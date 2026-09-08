"""Public interface of the db component.

`init_db` reads `config.toml`, opens the SQLite database, and creates the tables.
"""

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

from peewee import PeeweeException, SqliteDatabase

from db.schema import Image, Region, database_proxy

logger: logging.Logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler: logging.StreamHandler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s-%(asctime)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

DB_DIR: Path = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH: Path = DB_DIR / "config.toml"

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
