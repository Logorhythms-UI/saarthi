# DB

This is the db module.
It stores the metadata of the captured images in a SQLite database.
It uses the `peewee` ORM.

The module does not store the embedding vectors.
The vectors are in a FAISS index.
The `Region.id` value is the vector id in the FAISS index.

For the full design, see `docs/db/Ingestion Database Schema.md`.

## Files

| File | Purpose |
|------|---------|
| `schema.py` | The `Image` and `Region` tables |
| `api.py` | Config loader, `init_db`, the query helpers and the start-up recovery |
| `config.toml` | Paths of the SQLite file and the FAISS index |

## Rules

- This module does not control the flow of the application. The `backend` module does that.
- This module gives small helpers. Each helper does one database job.
- This module does not open the FAISS file. It does not read or write image files.

## Tables

- `image`: one row for each uploaded image.
- `region`: one row for each detected box in an image.

## Configuration

Edit `config.toml` to change where the data is stored.
All paths are relative to the `db/` folder.
Absolute paths and paths that leave the `db/` folder are not allowed.

```toml
[paths]
sqlite = "data/saarthi.db"
faiss_index = "data/regions.faiss"
```

## Usage

```python
from db.api import init_db

database, config = init_db()
```

`init_db` opens the database and creates the missing tables.
You can call it more than one time.
Use `config.faiss_path` to open the FAISS index.

## Query helpers

All helpers are in `api.py`. The helpers return `Image` and `Region` objects.

| Helper | Task |
|--------|------|
| `add_image` | Insert an image with the status `pending` |
| `find_image_by_sha256` | Find an image with the same hash (duplicate upload) |
| `get_image` | Read one image |
| `claim_next_pending` | Set the oldest `pending` image to `processing` |
| `set_image_analysis` | Save `phash` and `blur_score` of a `processing` image |
| `add_regions` | Insert the regions of a `processing` image |
| `get_regions` | Read the regions of an image |
| `mark_regions_embedded` | Record that the vectors are in the FAISS index |
| `mark_image_done` | Set a `processing` image to `done` |
| `mark_image_rejected` | Set a `processing` image to `rejected`, with a reason |
| `mark_image_failed` | Set a `processing` image to `failed`, with an error text |
| `list_embedded_region_ids` | List the region ids that have a vector |
| `recover_interrupted` | Repair the database after a stop |

A helper that changes the status of an image works only from the correct status.
If the status is not correct, the helper raises `InvalidStateError`.

## Errors

| Error | Meaning |
|-------|---------|
| `ConfigError` | `config.toml` is missing or not correct |
| `DbError` | Base class of the errors below. Also used for other database errors |
| `DuplicateImageError` | `file_path` or `sha256` is already in the database |
| `InvalidStateError` | The image does not exist or has the wrong status |
| `InvalidRecordError` | The database refuses a value |

## Start-up

1. Call `init_db()`.
2. Call `recover_interrupted()` before you start the workers.
3. Remove the ids in `stale_vector_ids` from the FAISS index.
4. Compare `list_embedded_region_ids()` with the ids in the FAISS index.
