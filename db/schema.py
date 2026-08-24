import datetime
from pathlib import Path

from peewee import (
    AutoField,
    DateTimeField,
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
