import datetime
from peewee import AutoField, DateTimeField, Model, SqliteDatabase, TextField

db = SqliteDatabase("database.db")


class BadeModel(Model):
    class Meta:
        database = db


class ImageHistory(BadeModel):
    id = AutoField()
    imgpath = TextField(unique=True, null=False)
    timestamp = DateTimeField(default=datetime.datetime.now)
