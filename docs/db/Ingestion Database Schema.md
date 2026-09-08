# Ingestion Database Schema

## Purpose

This document describes the database for the image ingestion pipeline.
The pipeline runs on a local Raspberry Pi server.
A client sends images one at a time.
The server saves each image on disk without its metadata (EXIF).
The server then records its own metadata in a SQLite database.

This design covers the ingestion data only.
The object gallery, the last-seen table and the sighting log are not in this design.

## Design rules

- The server stores the vectors in a FAISS index, not in SQLite.
- `Region.id` is the vector id in the FAISS index. This is the only link between the two stores.
- All timestamps are UTC.
- `captured_at` is the time at which the server received the image. It is not the time in the image metadata.
- The database refuses wrong values. `CHECK` constraints limit `status` and `source`, and they check the image size and the box coordinates.

## Table `image`

One row for each uploaded image.

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer, key | Image id |
| `file_path` | text, unique | Path of the file, relative to the storage root. Never absolute |
| `sha256` | text(64), unique | Hash of the stored file. It rejects exact duplicate uploads |
| `phash` | text(16), index, null | Perceptual hash for near-duplicate checks. Null until computed |
| `width` | integer | Width in pixels. Must be more than 0 |
| `height` | integer | Height in pixels. Must be more than 0 |
| `captured_at` | datetime, index | Time at which the server received the image |
| `room` | text, null | Name of the place |
| `blur_score` | float, null | Laplacian variance. Null until computed |
| `status` | text, index | Processing state. See below |
| `status_detail` | text, null | Reason for a reject or text of an error. It must not contain image content or personal data |
| `processed_at` | datetime, null | Time at which processing ended |

### Image status

| Value | Meaning |
|-------|---------|
| `pending` | The image is on disk. Nobody has analysed it yet |
| `processing` | A worker analyses the image |
| `done` | The regions and the vectors are written |
| `rejected` | A quality gate dropped the image (for example blur or duplicate) |
| `failed` | An error stopped the processing |

## Table `region`

One row for each detected box in an image.

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer, key | Region id. It is also the FAISS vector id |
| `image` | foreign key, index | Parent image. The database deletes the region when it deletes the image |
| `x0`, `y0` | integer | Top-left corner in pixels. Must be 0 or more |
| `x1`, `y1` | integer | Bottom-right corner in pixels. `x1` must be more than `x0`. `y1` must be more than `y0` |
| `det_score` | float | Detector score. Use it for analysis only. Do not use it to decide identity |
| `source` | text | Detector pass that made the box. See below |
| `phrases` | JSON list | Detector labels that remain after the merge of duplicate boxes |
| `in_hand` | boolean | True when a hand covers the object |
| `quality_ok` | boolean | False when the crop failed the size, blur or edge check |
| `embedding_model` | text, index, null | Name of the model that made the vector. Null means the vector is not in the FAISS index |

### Region source

| Value | Meaning |
|-------|---------|
| `full` | Full-frame pass |
| `tile` | Tiled pass for small objects |
| `generic` | Generic "an object" pass |

## Processing flow

1. The server saves the file and inserts an `image` row with the status `pending`.
2. A worker sets the status to `processing`.
3. The worker inserts the `region` rows. This gives the region ids.
4. The worker adds the vectors to the FAISS index, with the region ids as vector ids.
5. The worker sets `embedding_model` on each region.
6. The worker sets the image status to `done`.

If the server stops during step 4, some regions keep a null `embedding_model`.
A recovery job can find these regions and make their vectors again.

## Configuration

The file `db/config.toml` holds the paths of the SQLite file and the FAISS index.
The paths are relative to the `db/` folder.
The loader refuses absolute paths and paths that leave the `db/` folder.

| Key | Default | Description |
|-----|---------|-------------|
| `paths.sqlite` | `data/saarthi.db` | SQLite file |
| `paths.faiss_index` | `data/regions.faiss` | FAISS index file |

`init_db()` reads the file, opens the database, sets these SQLite options, and creates the missing tables:

| Option | Value | Reason |
|--------|-------|--------|
| `foreign_keys` | on | Enforce the link between region and image |
| `journal_mode` | `wal` | Readers do not block the writer |
| `busy_timeout` | 5000 ms | Wait for a lock before an error |
| `synchronous` | `normal` | Good speed with WAL |

The db component does not open the FAISS file.
The component that owns the FAISS index reads `faiss_path` from the config that `init_db()` returns.

## Logging

The module logs in this format: `{module}-{time}: {message}`.

| Level | Event |
|-------|-------|
| INFO | The config is loaded, the database is open, the tables are ready |
| ERROR | The config is not valid, or the database cannot open |

The logs do not contain image content or personal data.
