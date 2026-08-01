from peewee import SqliteDatabase, TextField, DateTimeField, Model, AutoField
import datetime

db = SqliteDatabase("database.db")


class BadeModel(Model):
    class Meta:
        database = db


class ImageHistory(BadeModel):
    img_name = AutoField()
    objects_identified = TextField(
        null=True, index=True
    )  # string list of objects identified with : as delimiter
    timestamp = DateTimeField(default=datetime.datetime.now)

