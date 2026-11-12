"""Settings of the storage component. They come from `config.toml`."""

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

from storage.errors import StorageConfigError

logger: logging.Logger = logging.getLogger(__name__)

STORAGE_DIR: Path = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH: Path = STORAGE_DIR / "config.toml"

MAX_JPEG_QUALITY: int = 95  # Pillow advises against higher values


@dataclass(frozen=True)
class StorageConfig:
    root: Path  # Absolute path of the folder for the image files
    max_bytes: int
    max_pixels: int
    min_free_bytes: int
    jpeg_quality: int
    retention_days: int
    orphan_grace_seconds: int


def _table(raw: dict, name: str) -> dict:
    table: object = raw.get(name)
    if not isinstance(table, dict):
        raise StorageConfigError(f"Missing [{name}] table in config")
    return table


def _integer(
    table: dict, name: str, key: str, minimum: int, maximum: int | None = None
) -> int:
    if key not in table:
        raise StorageConfigError(f"Missing key {name}.{key} in config")
    value: object = table[key]
    too_big: bool = maximum is not None and isinstance(value, int) and value > maximum
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or too_big
    ):
        limit: str = (
            f"between {minimum} and {maximum}" if maximum else f"{minimum} or more"
        )
        raise StorageConfigError(f"{name}.{key} must be an integer {limit}")
    return value


def _resolve_root(table: dict) -> Path:
    value: object = table.get("root")
    if not isinstance(value, str) or not value.strip():
        raise StorageConfigError("location.root must be a non-empty string")
    path: Path = Path(value)
    if path.is_absolute():
        return path.resolve()
    resolved: Path = (STORAGE_DIR / path).resolve()
    if not resolved.is_relative_to(STORAGE_DIR):
        raise StorageConfigError("location.root must stay inside the storage folder")
    return resolved


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> StorageConfig:
    """Read and check the config file. Raise `StorageConfigError` if it is not correct."""
    assert isinstance(config_path, Path), "config_path must be a Path"
    try:
        with config_path.open("rb") as file:
            raw: dict = tomllib.load(file)
    except FileNotFoundError as error:
        raise StorageConfigError(f"Config file not found: {config_path}") from error
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise StorageConfigError(
            f"Cannot read config {config_path}: {error}"
        ) from error

    location: dict = _table(raw, "location")
    limits: dict = _table(raw, "limits")
    image: dict = _table(raw, "image")
    retention: dict = _table(raw, "retention")
    cleanup: dict = _table(raw, "cleanup")

    config: StorageConfig = StorageConfig(
        root=_resolve_root(location),
        max_bytes=_integer(limits, "limits", "max_bytes", 1),
        max_pixels=_integer(limits, "limits", "max_pixels", 1),
        min_free_bytes=_integer(limits, "limits", "min_free_bytes", 0),
        jpeg_quality=_integer(image, "image", "jpeg_quality", 1, MAX_JPEG_QUALITY),
        retention_days=_integer(retention, "retention", "days", 0),
        orphan_grace_seconds=_integer(cleanup, "cleanup", "orphan_grace_seconds", 0),
    )
    logger.info("Loaded storage config from %s", config_path)
    return config
