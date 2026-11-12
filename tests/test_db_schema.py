from collections.abc import Iterator

import pytest
from peewee import IntegrityError, SqliteDatabase

from db.schema import (
    Image,
    ImageStatus,
    Region,
    RegionSource,
    database_proxy,
)


@pytest.fixture()
def database() -> Iterator[SqliteDatabase]:
    """Bind an in-memory database with foreign keys on, and create the tables."""
    db: SqliteDatabase = SqliteDatabase(":memory:", pragmas={"foreign_keys": 1})
    database_proxy.initialize(db)
    db.create_tables([Image, Region])
    yield db
    db.close()


def make_image(n: int = 1, **overrides: object) -> Image:
    fields: dict[str, object] = {
        "file_path": f"2026/10/{n}.jpg",
        "sha256": f"{n:064x}",
        "width": 640,
        "height": 480,
    }
    fields.update(overrides)
    return Image.create(**fields)


def make_region(image: Image, **overrides: object) -> Region:
    fields: dict[str, object] = {
        "image": image,
        "x0": 10,
        "y0": 20,
        "x1": 110,
        "y1": 120,
        "det_score": 0.31,
        "source": RegionSource.FULL.value,
    }
    fields.update(overrides)
    return Region.create(**fields)


def test_image_defaults(database: SqliteDatabase) -> None:
    image: Image = make_image()
    assert image.status == ImageStatus.PENDING.value
    assert image.captured_at is not None
    assert image.phash is None
    assert image.processed_at is None


def test_file_path_and_sha256_are_unique(database: SqliteDatabase) -> None:
    make_image(1)
    with pytest.raises(IntegrityError):
        make_image(2, file_path="2026/10/1.jpg")
    with pytest.raises(IntegrityError):
        make_image(3, sha256=f"{1:064x}")


def test_invalid_status_is_rejected(database: SqliteDatabase) -> None:
    with pytest.raises(IntegrityError):
        make_image(status="unknown")


def test_non_positive_size_is_rejected(database: SqliteDatabase) -> None:
    with pytest.raises(IntegrityError):
        make_image(width=0)


def test_region_round_trip(database: SqliteDatabase) -> None:
    region: Region = make_region(make_image(), phrases=["a bottle", "a flask"])
    loaded: Region = Region.get_by_id(region.id)
    assert loaded.phrases == ["a bottle", "a flask"]
    assert loaded.in_hand is False
    assert loaded.quality_ok is True
    assert loaded.embedding_model is None


def test_invalid_region_source_is_rejected(database: SqliteDatabase) -> None:
    with pytest.raises(IntegrityError):
        make_region(make_image(), source="other")


def test_invalid_box_is_rejected(database: SqliteDatabase) -> None:
    image: Image = make_image()
    with pytest.raises(IntegrityError):
        make_region(image, x1=5)  # x1 must be greater than x0
    with pytest.raises(IntegrityError):
        make_region(image, y0=-1)


def test_deleting_image_deletes_its_regions(database: SqliteDatabase) -> None:
    image: Image = make_image()
    make_region(image)
    make_region(image, x0=0)
    assert Region.select().count() == 2
    image.delete_instance()
    assert Region.select().count() == 0


def test_regions_without_vector_can_be_found(database: SqliteDatabase) -> None:
    image: Image = make_image()
    make_region(image)
    make_region(image, x0=1, embedding_model="dinov2-base")
    missing: list[Region] = list(
        Region.select().where(Region.embedding_model.is_null())
    )
    assert len(missing) == 1
