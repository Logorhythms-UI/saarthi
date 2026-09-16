"""Tests of the vector helpers in db.api. They use the actual database and index.

Each test makes its own rows and removes its own vectors and rows.
"""

import uuid
from collections.abc import Iterator

import numpy as np
import pytest

from db.api import (
    DbError,
    NewRegion,
    VectorIndexError,
    add_image,
    add_regions,
    checkpoint_vectors,
    clear_regions_embedded,
    get_regions,
    init_db,
    list_embedded_region_ids,
    remove_region_vectors,
    store_region_vectors,
    vector_count,
)
from db.schema import Image, ImageStatus, Region, RegionSource

TEST_PREFIX: str = "__test__/"
MODEL: str = "test-model"
DIMENSION: int = 384  # Same as vectors.dimension in config.toml


@pytest.fixture(scope="module", autouse=True)
def actual_database() -> Iterator[None]:
    database, config = init_db()
    assert config.vector_dimension == DIMENSION
    yield
    database.close()


@pytest.fixture(autouse=True)
def cleanup() -> Iterator[list[int]]:
    """Give the test a list. The test appends region ids that it stored."""
    stored: list[int] = []
    yield stored
    if stored:
        remove_region_vectors(stored)
    Image.delete().where(Image.file_path.startswith(TEST_PREFIX)).execute()


def make_regions(count: int) -> list[Region]:
    image: Image = add_image(
        f"{TEST_PREFIX}{uuid.uuid4().hex}.jpg", uuid.uuid4().hex * 2, 640, 480
    )
    Image.update(status=ImageStatus.PROCESSING.value).where(
        Image.id == image.id
    ).execute()
    new: list[NewRegion] = [
        NewRegion(i, i, i + 10, i + 10, 0.3, RegionSource.FULL, ["a bottle"])
        for i in range(count)
    ]
    return add_regions(image.id, new)


def unit_vectors(count: int, seed: int = 0) -> np.ndarray:
    rng: np.random.Generator = np.random.default_rng(seed)
    raw: np.ndarray = rng.normal(size=(count, DIMENSION)).astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def test_store_adds_vectors_and_marks_regions(cleanup: list[int]) -> None:
    before: int = vector_count()
    regions: list[Region] = make_regions(2)
    ids: list[int] = [r.id for r in regions]
    store_region_vectors(ids, unit_vectors(2), MODEL)
    cleanup.extend(ids)
    assert vector_count() == before + 2
    assert set(ids) <= set(list_embedded_region_ids())
    assert {r.embedding_model for r in get_regions(regions[0].image_id)} == {MODEL}


def test_store_rolls_back_the_index_if_the_database_step_fails() -> None:
    before: int = vector_count()
    missing_id: int = 2**31  # No region has this id
    with pytest.raises(DbError):
        store_region_vectors([missing_id], unit_vectors(1), MODEL)
    assert vector_count() == before


def test_store_refuses_an_id_that_is_already_stored(cleanup: list[int]) -> None:
    regions: list[Region] = make_regions(1)
    ids: list[int] = [regions[0].id]
    store_region_vectors(ids, unit_vectors(1), MODEL)
    cleanup.extend(ids)
    count: int = vector_count()
    with pytest.raises(VectorIndexError):
        store_region_vectors(ids, unit_vectors(1, seed=1), MODEL)
    assert vector_count() == count


def test_remove_clears_index_and_marks(cleanup: list[int]) -> None:
    regions: list[Region] = make_regions(2)
    ids: list[int] = [r.id for r in regions]
    store_region_vectors(ids, unit_vectors(2), MODEL)
    before: int = vector_count()
    assert remove_region_vectors([ids[0], 2**31]) == 1
    assert vector_count() == before - 1
    assert ids[0] not in list_embedded_region_ids()
    assert ids[1] in list_embedded_region_ids()
    cleanup.append(ids[1])


def test_clear_regions_embedded_ignores_unknown_ids() -> None:
    assert clear_regions_embedded([]) == 0
    assert clear_regions_embedded([2**31]) == 0


def test_checkpoint_writes_the_file() -> None:
    checkpoint_vectors()
