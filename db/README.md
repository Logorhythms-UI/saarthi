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
| `api.py` | `load_config` and `init_db` |
| `config.toml` | Paths of the SQLite file and the FAISS index |

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
