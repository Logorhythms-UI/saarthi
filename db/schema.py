import datetime
from pathlib import Path

from peewee import AutoField, DateTimeField, Model, SqliteDatabase, TextField

# Place database.db next to this file (inside db/) regardless of CWD
_db_path: str = str(Path(__file__).parent / "database.db")
db: SqliteDatabase = SqliteDatabase(_db_path)


class BaseModel(Model):
    class Meta:
        database = db


class ImageHistory(BaseModel):
    id = AutoField()
    imgpath = TextField(unique=True, null=False)
    timestamp = DateTimeField(default=datetime.datetime.now)
