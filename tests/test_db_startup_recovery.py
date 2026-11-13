"""Tests of the start-up recovery. They use the actual database and index.

`startup_recovery` repairs the whole database. A test skips if the actual
database or index already has a problem, so the test does not change real data.
"""

from collections.abc import Iterator

import pytest

from db.api import (
    StartupReport,
    get_image,
    get_regions,
    init_db,
    list_vector_ids,
    startup_recovery,
    store_region_vectors,
    vector_count,
)
from db.schema import Image, ImageStatus, Region
from tests.db_support import (
    MODEL,
    TEST_PREFIX,
    make_image,
    make_regions,
    set_status,
    skip_if_actual_data_has_problems,
    unit_vectors,
)

# Id that no region row has. Used for a vector without a region.
UNKNOWN_ID: int = 2**30


@pytest.fixture(scope="module", autouse=True)
def actual_database() -> Iterator[None]:
    database, _ = init_db()
    yield
    database.close()


@pytest.fixture(autouse=True)
def guard_and_cleanup() -> Iterator[None]:
    """Skip if the actual data has a problem. Remove the test data afterwards."""
    skip_if_actual_data_has_problems()
    yield
    leftover: list[int] = [
        r.id
        for r in Region.select(Region.id)
        .join(Image)
        .where(Image.file_path.startswith(TEST_PREFIX))
    ]
    # Reuse the recovery rule: a vector without a region is removed there.
    Image.delete().where(Image.file_path.startswith(TEST_PREFIX)).execute()
    startup_recovery()
    assert not set(leftover) & set(list_vector_ids())


def test_healthy_data_is_not_changed() -> None:
    image: Image = make_image()
    regions: list[Region] = make_regions(image, 2)
    store_region_vectors([r.id for r in regions], unit_vectors(2), MODEL)
    set_status(image, ImageStatus.DONE)
    before: int = vector_count()

    report: StartupReport = startup_recovery()

    assert report == StartupReport([], [], 0, 0)
    assert vector_count() == before
    assert get_image(image.id).status == ImageStatus.DONE.value
    assert len(get_regions(image.id)) == 2


def test_interrupted_image_is_reset_and_its_vectors_are_removed() -> None:
    image: Image = make_image()
    regions: list[Region] = make_regions(image, 2)
    store_region_vectors([regions[0].id], unit_vectors(1), MODEL)
    before: int = vector_count()

    report: StartupReport = startup_recovery()

    assert report.interrupted_image_ids == [image.id]
    assert get_image(image.id).status == ImageStatus.PENDING.value
    assert get_regions(image.id) == []
    assert vector_count() == before - 1
    assert regions[0].id not in list_vector_ids()


def test_orphan_vector_is_removed() -> None:
    from db.api import _require_index  # The test needs a vector with no region

    _require_index().add([UNKNOWN_ID], unit_vectors(1))

    report: StartupReport = startup_recovery()

    assert report.orphan_vectors_removed == 1
    assert UNKNOWN_ID not in list_vector_ids()


def test_missing_vector_clears_the_mark_and_requeues_the_image() -> None:
    image: Image = make_image()
    regions: list[Region] = make_regions(image, 2)
    store_region_vectors([r.id for r in regions], unit_vectors(2), MODEL)
    set_status(image, ImageStatus.DONE)
    from db.api import _require_index

    _require_index().remove([regions[0].id])  # Index lost one vector

    report: StartupReport = startup_recovery()

    assert report.missing_vectors_cleared == 1
    assert report.requeued_image_ids == [image.id]
    assert get_image(image.id).status == ImageStatus.PENDING.value
    assert get_regions(image.id) == []
    assert not {r.id for r in regions} & set(list_vector_ids())


def test_done_image_with_a_gap_is_requeued() -> None:
    image: Image = make_image()
    regions: list[Region] = make_regions(image, 2)
    store_region_vectors([regions[0].id], unit_vectors(1), MODEL)  # Second has none
    set_status(image, ImageStatus.DONE)

    report: StartupReport = startup_recovery()

    assert report.requeued_image_ids == [image.id]
    assert get_image(image.id).status == ImageStatus.PENDING.value
    assert get_image(image.id).processed_at is None
    assert regions[0].id not in list_vector_ids()


def test_second_call_has_nothing_to_repair() -> None:
    image: Image = make_image()
    make_regions(image, 1)  # processing image, no vector
    assert startup_recovery().interrupted_image_ids == [image.id]
    assert startup_recovery() == StartupReport([], [], 0, 0)
