"""Tests of the delete helpers. They use the actual database and index.

The test images get capture times in the year 2000, so the actual data (from
today) is never listed or deleted.
"""

from collections.abc import Iterator
from datetime import datetime

import pytest

from db.api import (
    InvalidStateError,
    delete_images,
    get_image,
    get_regions,
    init_db,
    list_embedded_region_ids,
    list_image_paths,
    list_images_older_than,
    list_vector_ids,
    remove_region_vectors,
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
    unit_vectors,
)

OLD: datetime = datetime(2000, 1, 1)
CUTOFF: datetime = datetime(2001, 1, 1)


@pytest.fixture(scope="module", autouse=True)
def actual_database() -> Iterator[None]:
    database, _ = init_db()
    yield
    database.close()


@pytest.fixture(autouse=True)
def cleanup() -> Iterator[None]:
    yield
    leftover: list[int] = [
        r.id
        for r in Region.select(Region.id)
        .join(Image)
        .where(Image.file_path.startswith(TEST_PREFIX))
    ]
    remove_region_vectors(leftover)
    Image.delete().where(Image.file_path.startswith(TEST_PREFIX)).execute()


def age(image: Image, status: ImageStatus, captured_at: datetime = OLD) -> Image:
    """Give an image a final status and an old capture time."""
    Image.update(status=status.value, captured_at=captured_at).where(
        Image.id == image.id
    ).execute()
    return image


def old_image(status: ImageStatus, captured_at: datetime = OLD) -> Image:
    return age(make_image(), status, captured_at)


def test_list_images_older_than_lists_only_finished_old_images_oldest_first() -> None:
    third: Image = old_image(ImageStatus.FAILED, datetime(2000, 3, 1))
    first: Image = old_image(ImageStatus.DONE, datetime(2000, 1, 1))
    second: Image = old_image(ImageStatus.REJECTED, datetime(2000, 2, 1))
    old_image(ImageStatus.PENDING)
    old_image(ImageStatus.PROCESSING)
    old_image(ImageStatus.DONE, datetime(2002, 1, 1))  # Newer than the cutoff
    result: list[Image] = list_images_older_than(CUTOFF)
    assert [i.id for i in result] == [first.id, second.id, third.id]
    assert [i.id for i in list_images_older_than(CUTOFF, limit=2)] == [
        first.id,
        second.id,
    ]
    assert list_images_older_than(datetime(1999, 1, 1)) == []


def test_delete_images_removes_rows_regions_and_vectors_and_returns_the_paths() -> None:
    image: Image = make_image()
    regions: list[Region] = make_regions(image, 2)  # Regions need a processing image
    ids: list[int] = [r.id for r in regions]
    store_region_vectors(ids, unit_vectors(2), MODEL)
    age(image, ImageStatus.DONE)
    other: Image = make_image()
    other_regions: list[Region] = make_regions(other, 1)
    store_region_vectors([other_regions[0].id], unit_vectors(1, seed=1), MODEL)
    age(other, ImageStatus.DONE)
    before: int = vector_count()

    paths: list[str] = delete_images([image.id, 2**31])  # The second id is unknown

    assert paths == [image.file_path]
    assert get_image(image.id) is None and get_regions(image.id) == []
    assert not set(ids) & set(list_vector_ids()) and not set(ids) & set(
        list_embedded_region_ids()
    )
    assert vector_count() == before - 2
    assert get_image(other.id) is not None and other_regions[0].id in list_vector_ids()


def test_an_image_in_use_stops_the_whole_delete() -> None:
    done: Image = old_image(ImageStatus.DONE)
    busy: Image = old_image(ImageStatus.DONE)
    set_status(busy, ImageStatus.PROCESSING)
    with pytest.raises(InvalidStateError):
        delete_images([done.id, busy.id])
    assert get_image(done.id) is not None and get_image(busy.id) is not None


def test_pending_images_can_be_deleted_by_id() -> None:
    pending: Image = old_image(ImageStatus.PENDING)
    assert delete_images([pending.id]) == [pending.file_path]


def test_delete_images_with_no_ids_does_nothing() -> None:
    assert delete_images([]) == []


def test_list_image_paths_is_sorted_and_has_every_path() -> None:
    first: Image = old_image(ImageStatus.DONE)
    second: Image = old_image(ImageStatus.PENDING)
    paths: list[str] = list_image_paths()
    assert paths == sorted(paths)
    assert {first.file_path, second.file_path} <= set(paths)


def test_parameters_are_checked() -> None:
    with pytest.raises(AssertionError):
        delete_images([0])
    with pytest.raises(AssertionError):
        delete_images("1")  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        list_images_older_than("2001")  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        list_images_older_than(CUTOFF, limit=0)
