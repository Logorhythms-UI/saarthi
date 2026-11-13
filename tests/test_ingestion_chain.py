"""The whole ingestion chain on one real photo, with the actual storage, db and index.

This is what the backend will do: upload, worker, duplicate check and retention.
It uses the real models, so it is slow.
Run it with: uv run pytest -m slow tests/test_ingestion_chain.py
"""

from collections.abc import Iterator
from datetime import timedelta

import numpy as np
import pytest

import storage.api as storage
from ai.api import AiEngine
from ai.types import Candidate
from db.api import (
    NewRegion,
    add_image,
    add_regions,
    claim_next_pending,
    delete_images,
    find_image_by_sha256,
    get_image,
    get_regions,
    init_db,
    list_image_paths,
    list_images_older_than,
    list_recent_phashes,
    list_vector_ids,
    mark_image_done,
    remove_region_vectors,
    set_image_analysis,
    startup_recovery,
    store_region_vectors,
)
from db.schema import Image, ImageStatus, RegionSource, utc_now
from tests.db_support import skip_if_actual_data_has_problems
from tests.image_support import TEST_IMAGE

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        TEST_IMAGE is None, reason="no test photo (set SAARTHI_TEST_IMAGE)"
    ),
]


@pytest.fixture(scope="module")
def engine() -> AiEngine:
    return AiEngine()


@pytest.fixture(autouse=True)
def actual_system() -> Iterator[list[str]]:
    """Open storage and db. Give the test a list for the file paths that it makes."""
    storage_config = storage.init_storage()
    database, db_config = init_db()
    assert db_config.vector_dimension > 0 and storage_config.root.is_dir()
    skip_if_actual_data_has_problems()
    if Image.select().where(Image.status == ImageStatus.PENDING.value).exists():
        pytest.skip(
            "Actual database has pending images. The chain would claim one of them"
        )
    made: list[str] = []
    yield made
    # Remove whatever the test left: files, vectors and rows.
    rows: list[Image] = list(Image.select().where(Image.file_path.in_(made)))
    if rows:
        remove_region_vectors([r.id for i in rows for r in i.regions])
        delete_images([i.id for i in rows])
    storage.delete_images(made)
    startup_recovery()
    database.close()


def to_new_region(candidate: Candidate) -> NewRegion:
    box = candidate.box
    return NewRegion(
        box.x0, box.y0, box.x1, box.y1, candidate.det_score,
        RegionSource(candidate.source), list(candidate.phrases),
        candidate.in_hand, candidate.quality_ok,
    )  # fmt: skip


def test_one_photo_goes_through_upload_worker_and_retention(
    engine: AiEngine, actual_system: list[str]
) -> None:
    upload: bytes = TEST_IMAGE.read_bytes()  # type: ignore[union-attr]

    # Upload: strip, check for a duplicate, save, add the row.
    prepared = storage.prepare_image(upload)
    assert find_image_by_sha256(prepared.sha256) is None
    path: str = storage.save_image(prepared)
    actual_system.append(path)
    row: Image = add_image(path, prepared.sha256, prepared.width, prepared.height)
    assert path in list_image_paths()
    assert storage.find_orphan_files(list_image_paths(), min_age_seconds=0) == []
    assert storage.find_missing_files([path]) == []

    # The same upload again has the same hash, so backend sees a duplicate.
    again = storage.prepare_image(upload)
    assert again.sha256 == prepared.sha256
    assert find_image_by_sha256(again.sha256).id == row.id  # type: ignore[union-attr]

    # Worker: read the stored file. It has the same pixel grid that the db knows.
    image = engine.open_image(storage.read_image(path))
    assert image.size == (prepared.width, prepared.height)
    claimed = claim_next_pending()
    assert claimed is not None and claimed.id == row.id
    report = engine.assess_quality(image)
    set_image_analysis(row.id, report.phash, report.blur_score)
    assert engine.embedding_dimension == init_db()[1].vector_dimension

    candidates: list[Candidate] = engine.detect_candidates(image)
    regions = add_regions(row.id, [to_new_region(c) for c in candidates])
    vectors: np.ndarray = engine.embed_regions(image, [c.box for c in candidates])
    if regions:
        store_region_vectors([r.id for r in regions], vectors, engine.embedding_model)
    mark_image_done(row.id)

    # The three stores agree.
    assert get_image(row.id).status == ImageStatus.DONE.value
    stored = get_regions(row.id)
    assert len(stored) == len(candidates)
    assert {r.id for r in stored} <= set(list_vector_ids())
    assert engine.is_near_duplicate(report.phash, list_recent_phashes())
    assert startup_recovery().requeued_image_ids == []

    # Retention: the image is "older than" a cutoff in the future. Delete only this one.
    expired = [
        i for i in list_images_older_than(utc_now() + timedelta(days=1), limit=10_000)
    ]
    assert row.id in [i.id for i in expired]
    paths: list[str] = delete_images([row.id])  # Rows, regions and vectors
    assert paths == [path]
    report_delete = storage.delete_images(paths)  # The file
    assert report_delete.deleted == [path]
    actual_system.clear()

    assert get_image(row.id) is None
    assert not {r.id for r in stored} & set(list_vector_ids())
    assert storage.find_missing_files([path]) == [path]
    result = startup_recovery()
    assert result.interrupted_image_ids == [] and result.requeued_image_ids == []
    assert result.orphan_vectors_removed == 0 and result.missing_vectors_cleared == 0
