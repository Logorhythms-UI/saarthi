"""Database tables for the image ingestion pipeline.

Two tables hold all ingestion data:
- `Image`: one row for each uploaded frame.
- `Region`: one row for each detected box in a frame (the "region archive").

`Region.id` is also the vector id in the FAISS index. The vectors are not
stored in SQLite.

All timestamps are naive UTC values.
"""

from datetime import UTC, datetime
from enum import StrEnum

from peewee import (
    BooleanField,
    CharField,
    Check,
    DatabaseProxy,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    TextField,
)
from playhouse.sqlite_ext import JSONField

# The real database is bound at start-up by `db.api.init_db`.
database_proxy: DatabaseProxy = DatabaseProxy()


class ImageStatus(StrEnum):
    """Processing state of an image."""

    PENDING = "pending"  # Saved on disk, not yet analysed
    PROCESSING = "processing"  # A worker is analysing the image
    DONE = "done"  # Regions and vectors are written
    REJECTED = "rejected"  # Dropped by a quality gate (blur, duplicate)
    FAILED = "failed"  # Processing raised an error


class RegionSource(StrEnum):
    """Detector pass that produced a region."""

    FULL = "full"  # Full-frame pass
    TILE = "tile"  # Tiled pass for small objects
    GENERIC = "generic"  # Generic "an object" pass


def _check_in(column: str, values: type[StrEnum]) -> Check:
    """Build a CHECK constraint that limits `column` to the enum values."""
    assert column.isidentifier(), f"Invalid column name: {column!r}"
    quoted: str = ", ".join(f"'{member.value}'" for member in values)
    return Check(f"{column} IN ({quoted})")


def utc_now() -> datetime:
    """Return the current UTC time as a naive datetime."""
    return datetime.now(UTC).replace(tzinfo=None)


class BaseModel(Model):
    """Base class that binds all tables to the shared database proxy."""

    class Meta:
        database = database_proxy


class Image(BaseModel):
    """One uploaded frame. The file on disk has no metadata (EXIF is stripped)."""

    # Relative to the storage root. Never absolute.
    file_path: CharField = CharField(unique=True)
    # Hash of the stripped file. Rejects exact duplicate uploads.
    sha256: CharField = CharField(max_length=64, unique=True)
    # Perceptual hash for near-duplicate checks. Null until computed.
    phash: CharField = CharField(max_length=16, null=True, index=True)
    width: IntegerField = IntegerField(constraints=[Check("width > 0")])
    height: IntegerField = IntegerField(constraints=[Check("height > 0")])
    # Time at which the server received the image.
    captured_at: DateTimeField = DateTimeField(default=utc_now, index=True)
    room: CharField = CharField(null=True)
    # Laplacian variance. Null until computed.
    blur_score: FloatField = FloatField(null=True)
    status: CharField = CharField(
        default=ImageStatus.PENDING.value,
        index=True,
        constraints=[_check_in("status", ImageStatus)],
    )
    # Reject reason or error text. Must not contain image content or personal data.
    status_detail: TextField = TextField(null=True)
    processed_at: DateTimeField = DateTimeField(null=True)

    class Meta:
        table_name = "image"


class Region(BaseModel):
    """One detected box in an image."""

    image: ForeignKeyField = ForeignKeyField(
        Image, backref="regions", on_delete="CASCADE", index=True
    )
    # Pixel coordinates in the frame of the parent image.
    x0: IntegerField = IntegerField(constraints=[Check("x0 >= 0")])
    y0: IntegerField = IntegerField(constraints=[Check("y0 >= 0")])
    x1: IntegerField = IntegerField(constraints=[Check("x1 > x0")])
    y1: IntegerField = IntegerField(constraints=[Check("y1 > y0")])
    # Detector score. For analysis only. Not used to decide identity.
    det_score: FloatField = FloatField()
    source: CharField = CharField(constraints=[_check_in("source", RegionSource)])
    # List of detector labels that remain after merging duplicate boxes.
    phrases: JSONField = JSONField(default=list)
    in_hand: BooleanField = BooleanField(default=False)
    # False when the crop failed the size, blur or edge gate.
    quality_ok: BooleanField = BooleanField(default=True)
    # Model that made the vector. Null means the vector is not in FAISS yet.
    embedding_model: CharField = CharField(null=True, index=True)

    class Meta:
        table_name = "region"
