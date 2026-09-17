"""The whole ingestion chain on one real photo, with the actual db and index.

This is what the backend worker will do. It uses the real models, so it is slow.
Run it with: uv run pytest -m slow tests/test_ingestion_chain.py
"""

import uuid
from collections.abc import Iterator

import numpy as np
import pytest

from ai.api import AiEngine
from ai.types import Candidate
from db.api import (
    NewRegion,
    add_image,
    add_regions,
    claim_next_pending,
    find_image_by_sha256,
    get_image,
    get_regions,
    init_db,
    list_recent_phashes,
    list_vector_ids,
    mark_image_done,
    remove_region_vectors,
    set_image_analysis,
    startup_recovery,
    store_region_vectors,
)
from db.schema import Image, ImageStatus, RegionSource
from tests.db_support import TEST_PREFIX, skip_if_actual_data_has_problems
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
def actual_database() -> Iterator[None]:
    database, config = init_db()
    assert config.vector_dimension > 0
    skip_if_actual_data_has_problems()
    if Image.select().where(Image.status == ImageStatus.PENDING.value).exists():
        pytest.skip(
            "Actual database has pending images. The chain would claim one of them"
        )
    yield
    ids = [
        r.id
        for i in Image.select().where(Image.file_path.startswith(TEST_PREFIX))
        for r in i.regions
    ]
    remove_region_vectors(ids)
    Image.delete().where(Image.file_path.startswith(TEST_PREFIX)).execute()
    startup_recovery()
    database.close()


def to_new_region(candidate: Candidate) -> NewRegion:
    box = candidate.box
    return NewRegion(
        box.x0, box.y0, box.x1, box.y1, candidate.det_score,
        RegionSource(candidate.source), list(candidate.phrases),
        candidate.in_hand, candidate.quality_ok,
    )  # fmt: skip


def test_one_photo_goes_through_the_whole_chain(engine: AiEngine) -> None:
    data: bytes = TEST_IMAGE.read_bytes()  # type: ignore[union-attr]
    image = engine.open_image(data)
    report = engine.assess_quality(image)
    assert engine.embedding_dimension == init_db()[1].vector_dimension

    # Stage 1: store the row, then claim it as a worker does.
    sha: str = uuid.uuid4().hex * 2
    row: Image = add_image(
        f"{TEST_PREFIX}{uuid.uuid4().hex}.jpg", sha, image.width, image.height
    )
    assert find_image_by_sha256(sha) is not None
    claimed: Image | None = claim_next_pending()
    assert claimed is not None and claimed.id == row.id
    set_image_analysis(row.id, report.phash, report.blur_score)
    # The blur gate is not tested here. Its threshold needs tuning (see the benchmark).
    assert report.blur_score > 0

    # Stage 2 and 3: detect, store the regions, embed, store the vectors.
    candidates: list[Candidate] = engine.detect_candidates(image)
    regions = add_regions(row.id, [to_new_region(c) for c in candidates])
    vectors: np.ndarray = engine.embed_regions(image, [c.box for c in candidates])
    if regions:
        store_region_vectors([r.id for r in regions], vectors, engine.embedding_model)
    mark_image_done(row.id)

    # The db and the index agree.
    assert get_image(row.id).status == ImageStatus.DONE.value
    stored = get_regions(row.id)
    assert len(stored) == len(candidates)
    assert {r.embedding_model for r in stored} <= {engine.embedding_model}
    assert {r.id for r in stored} <= set(list_vector_ids())
    assert report.phash in list_recent_phashes()
    assert engine.is_near_duplicate(report.phash, list_recent_phashes())

    # A start-up recovery finds nothing to repair.
    result = startup_recovery()
    assert result.interrupted_image_ids == [] and result.requeued_image_ids == []
    assert result.orphan_vectors_removed == 0 and result.missing_vectors_cleared == 0
