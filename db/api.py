"""Public interface of the db component.

`init_db` reads `config.toml`, opens the SQLite database, and creates the tables.
"""

import logging
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from peewee import IntegrityError, PeeweeException, SqliteDatabase

from db.schema import (
    Image,
    ImageStatus,
    Region,
    RegionSource,
    database_proxy,
    utc_now,
)

logger: logging.Logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler: logging.StreamHandler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s-%(asctime)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

DB_DIR: Path = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH: Path = DB_DIR / "config.toml"

# Longest `status_detail` text that the database stores.
MAX_STATUS_DETAIL_LENGTH: int = 500

_PATH_KEYS: tuple[str, ...] = ("sqlite", "faiss_index")
_SQLITE_PRAGMAS: dict[str, object] = {
    "foreign_keys": 1,
    "journal_mode": "wal",
    "busy_timeout": 5000,
    "synchronous": "normal",
}


class ConfigError(Exception):
    """Raised when `config.toml` is missing or has a wrong value."""


@dataclass(frozen=True)
class NewRegion:
    """Data for one region to insert. Coordinates are pixels in the image frame."""

    x0: int
    y0: int
    x1: int
    y1: int
    det_score: float
    source: RegionSource
    phrases: list[str]
    in_hand: bool = False
    quality_ok: bool = True

    def __post_init__(self) -> None:
        assert all(isinstance(v, int) for v in (self.x0, self.y0, self.x1, self.y1))
        assert 0 <= self.x0 < self.x1, "x1 must be more than x0, and x0 at least 0"
        assert 0 <= self.y0 < self.y1, "y1 must be more than y0, and y0 at least 0"
        assert isinstance(self.det_score, float), "det_score must be a float"
        assert isinstance(self.source, RegionSource), "source must be a RegionSource"
        assert isinstance(self.phrases, list)
        assert all(isinstance(phrase, str) for phrase in self.phrases)


@dataclass(frozen=True)
class RecoveryReport:
    """Result of `recover_interrupted`. The backend acts on the FAISS index."""

    # Images that were `processing` and are `pending` again.
    reset_image_ids: list[int]
    # Deleted regions that had a vector. The backend must remove these ids from FAISS.
    stale_vector_ids: list[int]
    # Regions of `done` images with no vector. The backend must decide what to do.
    unembedded_region_ids: list[int]


class DbError(Exception):
    """Base class of the errors that the query helpers raise."""


class DuplicateImageError(DbError):
    """Raised when `file_path` or `sha256` of an image already exists."""


class InvalidStateError(DbError):
    """Raised when an image does not exist or has the wrong status."""


class InvalidRecordError(DbError):
    """Raised when the database refuses a value (for example a bad box)."""


@dataclass(frozen=True)
class DbConfig:
    """Resolved settings of the db component."""

    sqlite_path: Path
    faiss_path: Path


def _resolve_inside_db_dir(key: str, value: object) -> Path:
    """Turn a relative config path into an absolute path inside the db folder."""
    if not isinstance(value, str) or not value:
        raise ConfigError(f"paths.{key} must be a non-empty string")
    if Path(value).is_absolute():
        raise ConfigError(f"paths.{key} must be relative to the db folder")
    resolved: Path = (DB_DIR / value).resolve()
    if not resolved.is_relative_to(DB_DIR):
        raise ConfigError(f"paths.{key} must stay inside the db folder")
    return resolved


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> DbConfig:
    """Read and validate the config file."""
    assert isinstance(config_path, Path), "config_path must be a Path"
    try:
        with config_path.open("rb") as file:
            raw: dict = tomllib.load(file)
    except FileNotFoundError as error:
        raise ConfigError(f"Config file not found: {config_path}") from error
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"Cannot read config {config_path}: {error}") from error

    paths: object = raw.get("paths")
    if not isinstance(paths, dict):
        raise ConfigError("Missing [paths] table in config")
    for key in _PATH_KEYS:
        if key not in paths:
            raise ConfigError(f"Missing key paths.{key} in config")

    config: DbConfig = DbConfig(
        sqlite_path=_resolve_inside_db_dir("sqlite", paths["sqlite"]),
        faiss_path=_resolve_inside_db_dir("faiss_index", paths["faiss_index"]),
    )
    logger.info("Loaded config from %s", config_path)
    return config


def init_db(config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[SqliteDatabase, DbConfig]:
    """Open the database named in the config and create missing tables.

    Return the database and the resolved config. The caller that owns the
    FAISS index reads `faiss_path` from the config. This function is safe to
    call more than once.
    """
    assert isinstance(config_path, Path), "config_path must be a Path"
    try:
        config: DbConfig = load_config(config_path)
    except ConfigError as error:
        logger.error("Invalid config: %s", error)
        raise

    try:
        config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        database: SqliteDatabase = SqliteDatabase(
            str(config.sqlite_path), pragmas=_SQLITE_PRAGMAS
        )
        database_proxy.initialize(database)
        database.connect(reuse_if_open=True)
        logger.info("Opened database %s", config.sqlite_path)
        database.create_tables([Image, Region], safe=True)
    except (OSError, PeeweeException) as error:
        logger.error(
            "Cannot open database %s: %s: %s",
            config.sqlite_path,
            type(error).__name__,
            error,
        )
        raise
    logger.info("Tables ready: image, region")
    return database, config


@contextmanager
def _translate_errors(action: str) -> Iterator[None]:
    """Log peewee errors and raise them as the typed errors of this module."""
    try:
        yield
    except IntegrityError as error:
        logger.error("%s failed: %s: %s", action, type(error).__name__, error)
        if "UNIQUE" in str(error).upper():
            raise DuplicateImageError(f"{action}: value already exists") from error
        raise InvalidRecordError(f"{action}: database refused the value") from error
    except PeeweeException as error:
        logger.error("%s failed: %s: %s", action, type(error).__name__, error)
        raise DbError(f"{action}: database error") from error


def add_image(
    file_path: str, sha256: str, width: int, height: int, room: str | None = None
) -> Image:
    """Insert a new image with the status `pending`.

    `file_path` is relative to the storage root. `captured_at` is the time at
    which this function runs. Raise `DuplicateImageError` if `file_path` or
    `sha256` is already in the database.
    """
    assert isinstance(file_path, str) and file_path, "file_path must be a string"
    assert not file_path.startswith(("/", "\\")), "file_path must be relative"
    assert ".." not in PurePosixPath(file_path).parts, "file_path must not use .."
    assert isinstance(sha256, str) and len(sha256) == 64, "sha256 must have 64 chars"
    assert isinstance(width, int) and width > 0, "width must be a positive int"
    assert isinstance(height, int) and height > 0, "height must be a positive int"
    assert room is None or isinstance(room, str), "room must be a string or None"

    with _translate_errors("add_image"):
        image: Image = Image.create(
            file_path=file_path,
            sha256=sha256,
            width=width,
            height=height,
            room=room,
            captured_at=utc_now(),
        )
    logger.debug("Added image %d with status pending", image.id)
    return image


def find_image_by_sha256(sha256: str) -> Image | None:
    """Return the image with this hash, or None. Use it to detect a duplicate upload."""
    assert isinstance(sha256, str) and len(sha256) == 64, "sha256 must have 64 chars"
    with _translate_errors("find_image_by_sha256"):
        return Image.get_or_none(Image.sha256 == sha256)


def get_image(image_id: int) -> Image | None:
    """Return the image with this id, or None."""
    assert isinstance(image_id, int) and image_id > 0, "image_id must be positive"
    with _translate_errors("get_image"):
        return Image.get_or_none(Image.id == image_id)


def claim_next_pending() -> Image | None:
    """Set the oldest `pending` image to `processing` and return it.

    The read and the update run in one write transaction. Two workers cannot
    get the same image. Return None when no image is `pending`.
    """
    with _translate_errors("claim_next_pending"), database_proxy.atomic("IMMEDIATE"):
        image: Image | None = (
            Image.select()
            .where(Image.status == ImageStatus.PENDING.value)
            .order_by(Image.captured_at, Image.id)
            .first()
        )
        if image is None:
            return None
        changed: int = (
            Image.update(status=ImageStatus.PROCESSING.value)
            .where(
                Image.id == image.id,
                Image.status == ImageStatus.PENDING.value,
            )
            .execute()
        )
        assert changed == 1, "claim must change exactly one row"
        image.status = ImageStatus.PROCESSING.value
    logger.info("Claimed image %d for processing", image.id)
    return image


def _finish_image(image_id: int, status: ImageStatus, detail: str | None) -> Image:
    """Move an image from `processing` to a final status."""
    assert isinstance(image_id, int) and image_id > 0, "image_id must be positive"
    assert detail is None or isinstance(detail, str), "detail must be a string"
    assert status in (ImageStatus.DONE, ImageStatus.REJECTED, ImageStatus.FAILED)

    cut_detail: str | None = (
        None if detail is None else detail[:MAX_STATUS_DETAIL_LENGTH]
    )
    with _translate_errors(f"mark image {status.value}"):
        changed: int = (
            Image.update(
                status=status.value, status_detail=cut_detail, processed_at=utc_now()
            )
            .where(
                Image.id == image_id,
                Image.status == ImageStatus.PROCESSING.value,
            )
            .execute()
        )
        if changed != 1:
            raise InvalidStateError(
                f"Image {image_id} does not exist or is not in status processing"
            )
        image: Image = Image.get_by_id(image_id)
    logger.info("Image %d is now %s", image_id, status.value)
    return image


def mark_image_done(image_id: int) -> Image:
    """Set a `processing` image to `done`."""
    return _finish_image(image_id, ImageStatus.DONE, None)


def mark_image_rejected(image_id: int, reason: str) -> Image:
    """Set a `processing` image to `rejected`. `reason` must not hold personal data."""
    assert isinstance(reason, str) and reason, "reason must be a non-empty string"
    return _finish_image(image_id, ImageStatus.REJECTED, reason)


def mark_image_failed(image_id: int, error: str) -> Image:
    """Set a `processing` image to `failed`. `error` must not hold personal data."""
    assert isinstance(error, str) and error, "error must be a non-empty string"
    return _finish_image(image_id, ImageStatus.FAILED, error)


def _require_processing(image_id: int) -> None:
    """Raise `InvalidStateError` unless the image exists and is `processing`."""
    is_processing: bool = (
        Image.select()
        .where(
            Image.id == image_id,
            Image.status == ImageStatus.PROCESSING.value,
        )
        .exists()
    )
    if not is_processing:
        raise InvalidStateError(
            f"Image {image_id} does not exist or is not in status processing"
        )


def set_image_analysis(image_id: int, phash: str, blur_score: float) -> Image:
    """Save the quality-gate values of a `processing` image."""
    assert isinstance(image_id, int) and image_id > 0, "image_id must be positive"
    assert isinstance(phash, str) and 0 < len(phash) <= 16, "phash: 1 to 16 chars"
    assert isinstance(blur_score, float) and blur_score >= 0, "blur_score must be >= 0"

    with _translate_errors("set_image_analysis"):
        changed: int = (
            Image.update(phash=phash, blur_score=blur_score)
            .where(
                Image.id == image_id,
                Image.status == ImageStatus.PROCESSING.value,
            )
            .execute()
        )
        if changed != 1:
            raise InvalidStateError(
                f"Image {image_id} does not exist or is not in status processing"
            )
        return Image.get_by_id(image_id)


def add_regions(image_id: int, regions: list[NewRegion]) -> list[Region]:
    """Insert the regions of a `processing` image in one transaction.

    Return the new rows, with ids set and `embedding_model` empty. The ids are
    the vector ids for the FAISS index. An empty list is allowed.
    """
    assert isinstance(image_id, int) and image_id > 0, "image_id must be positive"
    assert isinstance(regions, list), "regions must be a list"
    assert all(isinstance(region, NewRegion) for region in regions)

    with _translate_errors("add_regions"), database_proxy.atomic():
        _require_processing(image_id)
        rows: list[Region] = [
            Region.create(
                image=image_id,
                x0=region.x0,
                y0=region.y0,
                x1=region.x1,
                y1=region.y1,
                det_score=region.det_score,
                source=region.source.value,
                phrases=region.phrases,
                in_hand=region.in_hand,
                quality_ok=region.quality_ok,
            )
            for region in regions
        ]
    logger.debug("Added %d region(s) to image %d", len(rows), image_id)
    return rows


def get_regions(image_id: int) -> list[Region]:
    """Return the regions of an image, ordered by id."""
    assert isinstance(image_id, int) and image_id > 0, "image_id must be positive"
    with _translate_errors("get_regions"):
        return list(Region.select().where(Region.image == image_id).order_by(Region.id))


def mark_regions_embedded(region_ids: list[int], embedding_model: str) -> int:
    """Record that the vectors of these regions are in the FAISS index.

    Call it after the vectors are written. Change all rows or none: if one id
    does not exist, raise `InvalidStateError` and change nothing.
    """
    assert isinstance(region_ids, list) and region_ids, "region_ids must not be empty"
    assert all(isinstance(i, int) and i > 0 for i in region_ids), "ids must be positive"
    assert isinstance(embedding_model, str) and embedding_model, "model name needed"

    unique_ids: set[int] = set(region_ids)
    with _translate_errors("mark_regions_embedded"), database_proxy.atomic():
        changed: int = (
            Region.update(embedding_model=embedding_model)
            .where(Region.id.in_(unique_ids))
            .execute()
        )
        if changed != len(unique_ids):
            raise InvalidStateError(
                f"{len(unique_ids) - changed} region id(s) do not exist"
            )
    logger.debug("Marked %d region(s) as embedded", changed)
    return changed


def list_embedded_region_ids() -> list[int]:
    """Return the ids of all regions that have a vector in the FAISS index.

    The backend compares this list with the ids in FAISS to find differences.
    """
    with _translate_errors("list_embedded_region_ids"):
        query = (
            Region.select(Region.id)
            .where(Region.embedding_model.is_null(False))
            .order_by(Region.id)
        )
        return [row.id for row in query]


def recover_interrupted() -> RecoveryReport:
    """Repair the database after a stop. Call it once at start-up.

    Run it after `init_db` and before the workers start. It changes the
    database only. The backend removes `stale_vector_ids` from the FAISS index.

    1. Each `processing` image goes back to `pending`.
    2. The regions of these images are deleted, so the retry cannot make
       duplicates. The ids of those that had a vector are reported.
    3. Regions of `done` images with no vector are reported and not changed.
    """
    with _translate_errors("recover_interrupted"), database_proxy.atomic("IMMEDIATE"):
        interrupted: list[int] = [
            row.id
            for row in Image.select(Image.id)
            .where(Image.status == ImageStatus.PROCESSING.value)
            .order_by(Image.id)
        ]
        stale_vector_ids: list[int] = []
        if interrupted:
            stale_vector_ids = [
                row.id
                for row in Region.select(Region.id)
                .where(
                    Region.image.in_(interrupted),
                    Region.embedding_model.is_null(False),
                )
                .order_by(Region.id)
            ]
            Region.delete().where(Region.image.in_(interrupted)).execute()
            Image.update(
                status=ImageStatus.PENDING.value,
                status_detail=None,
                processed_at=None,
            ).where(Image.id.in_(interrupted)).execute()
        unembedded: list[int] = [
            row.id
            for row in Region.select(Region.id)
            .join(Image)
            .where(
                Image.status == ImageStatus.DONE.value,
                Region.embedding_model.is_null(),
            )
            .order_by(Region.id)
        ]

    report: RecoveryReport = RecoveryReport(interrupted, stale_vector_ids, unembedded)
    logger.info(
        "Recovery: %d image(s) reset, %d stale vector id(s), %d region(s) without vector",
        len(interrupted),
        len(stale_vector_ids),
        len(unembedded),
    )
    if stale_vector_ids or unembedded:
        logger.warning(
            "Recovery needs action by backend: remove %d id(s) from FAISS, "
            "check %d region(s) without vector",
            len(stale_vector_ids),
            len(unembedded),
        )
    return report
