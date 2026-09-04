import datetime
from pathlib import Path

from peewee import (
    AutoField,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
    TextField,
)

# Place database.db next to this file (inside db/) regardless of CWD
_db_path: str = str(Path(__file__).parent / "database.db")
db: SqliteDatabase = SqliteDatabase(
    _db_path,
    pragmas={"journal_mode": "wal", "foreign_keys": 1},
    check_same_thread=False,
)


class BaseModel(Model):
    class Meta:
        database = db


class ImageHistory(BaseModel):
    id = AutoField()
    imgpath = TextField(unique=True, null=False)
    timestamp = DateTimeField(default=datetime.datetime.now)
    sequence_number = IntegerField(default=0)


class ObjectObservation(BaseModel):
    """A single detected object from a broad-detector pass on an ImageHistory image.

    Embeddings are NOT stored in SQLite — they live in FAISS indexes keyed by
    this record's integer id.  The bounding_box field holds a JSON-serialized
    list [x1, y1, x2, y2] in absolute pixel coordinates.
    """

    id = AutoField()
    image = ForeignKeyField(ImageHistory, backref="observations", on_delete="CASCADE")
    label = TextField()
    detector_confidence = FloatField()
    # JSON-serialized [x1, y1, x2, y2] in absolute pixel coordinates
    bounding_box = TextField()
    # Denormalized from parent image for fast temporal access during retrieval
    timestamp = DateTimeField()
