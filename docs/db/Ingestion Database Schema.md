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

The `backend` module controls this flow. The db module gives the helper for each step.

| Step | Action | Helper |
|------|--------|--------|
| 1 | Check for a duplicate upload | `find_image_by_sha256` |
| 2 | Save the file, then insert the image with the status `pending` | `add_image` |
| 3 | A worker takes the oldest `pending` image. The status becomes `processing` | `claim_next_pending` |
| 4 | The worker saves the quality-gate values | `set_image_analysis` |
| 5 | The worker inserts the regions. This gives the region ids | `add_regions` |
| 6 | The worker adds the vectors to the FAISS index, with the region ids as vector ids | (FAISS, not in the db module) |
| 7 | The worker sets `embedding_model` on each region | `mark_regions_embedded` |
| 8 | The worker sets the final status | `mark_image_done`, `mark_image_rejected` or `mark_image_failed` |

If the server stops during step 6, some regions keep a null `embedding_model`.
The start-up recovery finds them (see below).

## Query helpers

All helpers are in `db/api.py`. They return `Image` and `Region` objects.
Each helper checks its parameters with `assert`.
Each helper uses peewee expressions only. No helper builds SQL from text.

| Helper | Task | Result |
|--------|------|--------|
| `add_image(file_path, sha256, width, height, room=None)` | Insert an image with the status `pending`. `captured_at` is the time of the call. The path must be relative and must not contain `..` | `Image`. Raises `DuplicateImageError` |
| `find_image_by_sha256(sha256)` | Find an image by hash | `Image` or `None` |
| `get_image(image_id)` | Read one image | `Image` or `None` |
| `claim_next_pending()` | In one write transaction, set the oldest `pending` image to `processing`. Two workers cannot get the same image | `Image` or `None` |
| `set_image_analysis(image_id, phash, blur_score)` | Save the quality-gate values | `Image` |
| `add_regions(image_id, regions)` | Insert all regions in one transaction. An empty list is allowed | List of `Region` |
| `get_regions(image_id)` | Read the regions, ordered by id | List of `Region` |
| `mark_regions_embedded(region_ids, embedding_model)` | Record that the vectors are in FAISS. The helper changes all rows or none | Number of rows changed |
| `mark_image_done(image_id)` | Set the status `done` | `Image` |
| `mark_image_rejected(image_id, reason)` | Set the status `rejected` | `Image` |
| `mark_image_failed(image_id, error)` | Set the status `failed` | `Image` |
| `list_embedded_region_ids()` | List the ids of the regions that have a vector | List of integers |
| `recover_interrupted()` | Repair the database after a stop | `RecoveryReport` |

`add_regions` takes `NewRegion` objects. A `NewRegion` checks its own box:
`x0` must be 0 or more and less than `x1`, and `y0` must be 0 or more and less than `y1`.

These helpers work only on an image with the status `processing`:
`set_image_analysis`, `add_regions`, `mark_image_done`, `mark_image_rejected` and `mark_image_failed`.
Each of them changes the row with one `UPDATE` that tests the status.
If the status is not `processing`, the helper raises `InvalidStateError`.

The helpers `mark_image_rejected` and `mark_image_failed` keep the text in `status_detail`.
The helper cuts the text to 500 characters.
Do not put image content or personal data in this text.

### Errors

| Error | Meaning |
|-------|---------|
| `ConfigError` | `config.toml` is missing or not correct |
| `DbError` | Base class of the errors below. The helpers also raise it for other database errors |
| `DuplicateImageError` | `file_path` or `sha256` is already in the database |
| `InvalidStateError` | The image does not exist or has the wrong status. Also used when a region id does not exist |
| `InvalidRecordError` | The database refuses a value |

## Start-up recovery

The `backend` module calls `recover_interrupted()` one time at start.
It must call it after `init_db()` and before it starts the workers.
The helper changes the database only. It does not open the FAISS index.

The helper does these steps in one write transaction:

1. It finds the images with the status `processing`. A stop happened while a worker held them.
2. It collects the ids of their regions that have a vector.
3. It deletes all regions of these images. The retry makes new regions, so duplicates cannot occur.
4. It sets these images to `pending` and clears `status_detail` and `processed_at`.
5. It finds the regions of `done` images that have no vector. It does not change them.

The helper returns a `RecoveryReport`:

| Field | Meaning | Action for the backend |
|-------|---------|------------------------|
| `reset_image_ids` | Images that are `pending` again | None. The workers pick them up |
| `stale_vector_ids` | Deleted regions that had a vector | Remove these ids from the FAISS index |
| `unembedded_region_ids` | Regions of `done` images with no vector | Decide: make the vectors again, or log and ignore |

The db module cannot see the FAISS index.
The backend must also find the opposite error: a vector in FAISS with no region row.
Use `list_embedded_region_ids()` and compare the result with the ids in the FAISS index.

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
| DEBUG | An image or a group of regions is added, regions are marked as embedded |
| INFO | The config is loaded, the database is open, the tables are ready, an image is claimed, an image gets a final status, the recovery summary |
| WARNING | The recovery found work for the backend (stale vectors or regions without a vector) |
| ERROR | The config is not valid, the database cannot open, or a database call fails |

The default log level is INFO. DEBUG lines are not shown.

The logs do not contain image content or personal data.
