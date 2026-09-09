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

from db.schema import Image, ImageStatus, Region, database_proxy, utc_now

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
