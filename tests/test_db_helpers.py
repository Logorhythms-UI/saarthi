"""Tests of the query helpers. They run on the actual database from config.toml.

Every test makes its own rows with the path prefix `__test__/` and removes them.
"""

import uuid
from collections.abc import Iterator

import pytest

from db.api import (
    DuplicateImageError,
    InvalidStateError,
    add_image,
    claim_next_pending,
    find_image_by_sha256,
    get_image,
    init_db,
    mark_image_done,
    mark_image_failed,
    mark_image_rejected,
)
from db.schema import Image, ImageStatus

TEST_PREFIX: str = "__test__/"


@pytest.fixture(scope="module", autouse=True)
def actual_database() -> Iterator[None]:
    """Open the actual database. Remove leftover test rows before and after."""
    database, _ = init_db()
    Image.delete().where(Image.file_path.startswith(TEST_PREFIX)).execute()
    yield
    Image.delete().where(Image.file_path.startswith(TEST_PREFIX)).execute()
    database.close()


@pytest.fixture(autouse=True)
def clean_test_rows() -> Iterator[None]:
    yield
    Image.delete().where(Image.file_path.startswith(TEST_PREFIX)).execute()


def new_image(**overrides: object) -> Image:
    fields: dict[str, object] = {
        "file_path": f"{TEST_PREFIX}{uuid.uuid4().hex}.jpg",
        "sha256": uuid.uuid4().hex + uuid.uuid4().hex,
        "width": 640,
        "height": 480,
    }
    fields.update(overrides)
    return add_image(**fields)


def set_status(image: Image, status: ImageStatus) -> None:
    Image.update(status=status.value).where(Image.id == image.id).execute()


def skip_if_real_rows(status: ImageStatus) -> None:
    """Skip a test that would act on real rows with this status."""
    real: int = (
        Image.select()
        .where(Image.status == status.value, ~Image.file_path.startswith(TEST_PREFIX))
        .count()
    )
    if real:
        pytest.skip(f"Actual database has {real} real '{status.value}' image(s)")


def test_add_image_defaults() -> None:
    image: Image = new_image(room="kitchen")
    loaded: Image | None = get_image(image.id)
    assert loaded is not None
    assert loaded.status == ImageStatus.PENDING.value
    assert loaded.room == "kitchen"
    assert loaded.captured_at is not None


def test_duplicate_file_path_and_sha256_raise() -> None:
    first: Image = new_image()
    with pytest.raises(DuplicateImageError):
        new_image(file_path=first.file_path)
    with pytest.raises(DuplicateImageError):
        new_image(sha256=first.sha256)


@pytest.mark.parametrize(
    "overrides",
    [
        {"file_path": "/abs/a.jpg"},
        {"file_path": "__test__/../a.jpg"},
        {"sha256": "short"},
        {"width": 0},
        {"height": -1},
    ],
)
def test_add_image_rejects_bad_parameters(overrides: dict[str, object]) -> None:
    with pytest.raises(AssertionError):
        new_image(**overrides)


def test_find_image_by_sha256() -> None:
    image: Image = new_image()
    found: Image | None = find_image_by_sha256(image.sha256)
    assert found is not None and found.id == image.id
    assert find_image_by_sha256("0" * 64) is None


def test_get_image_missing_returns_none() -> None:
    assert get_image(2**31) is None


def test_claim_next_pending_takes_oldest_once() -> None:
    skip_if_real_rows(ImageStatus.PENDING)
    assert claim_next_pending() is None
    first: Image = new_image()
    second: Image = new_image()
    claimed: Image | None = claim_next_pending()
    assert claimed is not None and claimed.id == first.id
    assert claimed.status == ImageStatus.PROCESSING.value
    again: Image | None = claim_next_pending()
    assert again is not None and again.id == second.id
    assert claim_next_pending() is None


@pytest.mark.parametrize(
    ("mark", "argument", "status"),
    [
        (mark_image_done, None, ImageStatus.DONE),
        (mark_image_rejected, "too blurry", ImageStatus.REJECTED),
        (mark_image_failed, "model error", ImageStatus.FAILED),
    ],
)
def test_mark_image_from_processing(mark, argument, status: ImageStatus) -> None:
    image: Image = new_image()
    set_status(image, ImageStatus.PROCESSING)
    finished: Image = mark(image.id) if argument is None else mark(image.id, argument)
    assert finished.status == status.value
    assert finished.processed_at is not None
    assert finished.status_detail == argument


def test_mark_image_needs_processing_status() -> None:
    image: Image = new_image()  # status is pending
    with pytest.raises(InvalidStateError):
        mark_image_done(image.id)
    with pytest.raises(InvalidStateError):
        mark_image_done(2**31)


def test_status_detail_is_cut() -> None:
    image: Image = new_image()
    set_status(image, ImageStatus.PROCESSING)
    finished: Image = mark_image_failed(image.id, "x" * 5000)
    assert len(finished.status_detail) == 500
