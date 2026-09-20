"""Files on disk: safe paths, atomic writes, reads, deletes and checks.

All paths that go in or out of this module are relative to the storage root
and have the form `YYYY/MM/DD/<32 hex characters>.jpg`. Every path is checked
before it touches the disk.
"""

import errno
import logging
import os
import re
import shutil
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from storage.errors import (
    StorageFullError,
    StorageIOError,
    StorageNotFoundError,
    StoragePathError,
)
from storage.settings import StorageConfig

logger: logging.Logger = logging.getLogger(__name__)

# The only form of path that this module accepts. It has no "..", no
# separator tricks and no user-chosen name.
PATH_PATTERN: re.Pattern[str] = re.compile(r"\d{4}/\d{2}/\d{2}/[0-9a-f]{32}\.jpg")
TEMP_SUFFIX: str = ".tmp"
FILE_MODE: int = 0o600
FOLDER_MODE: int = 0o700
NAME_ATTEMPTS: int = 5


@dataclass(frozen=True)
class DeleteReport:
    """Result of `delete_files`."""

    deleted: list[str]
    missing: list[str]  # There was no file. Nothing to do
    failed: list[str]  # The disk refused. The file may still exist


def resolve_inside_root(config: StorageConfig, relative_path: str) -> Path:
    """Return the real path of a stored file. Raise `StoragePathError` if it is not safe."""
    if not isinstance(relative_path, str) or not PATH_PATTERN.fullmatch(relative_path):
        raise StoragePathError("The path is not a valid storage path")
    candidate: Path = (config.root / relative_path).resolve()
    # Defence in depth: a symbolic link must not lead out of the root.
    if not candidate.is_relative_to(config.root.resolve()):
        raise StoragePathError("The path leads out of the storage root")
    return candidate


def ensure_root(config: StorageConfig) -> None:
    """Create the root folder if needed and check that it can be written."""
    try:
        config.root.mkdir(parents=True, exist_ok=True, mode=FOLDER_MODE)
        if not config.root.is_dir():
            raise StorageIOError(f"The storage root is not a folder: {config.root}")
        probe: Path = config.root / f".write-test-{uuid.uuid4().hex}{TEMP_SUFFIX}"
        probe.write_bytes(b"x")
        probe.unlink()
    except OSError as error:
        logger.error("Cannot use the storage root %s: %s", config.root, error)
        raise StorageIOError(
            f"Cannot write to the storage root: {config.root}"
        ) from error


def free_space(config: StorageConfig) -> int:
    """Return the free space on the disk of the root, in bytes."""
    return shutil.disk_usage(config.root).free


def _make_folders(config: StorageConfig, relative_folder: str) -> Path:
    """Create the year, month and day folders. Only the new folders get mode 0700."""
    path: Path = config.root
    for part in relative_folder.split("/"):
        path = path / part
        if not path.exists():
            path.mkdir(mode=FOLDER_MODE)
            path.chmod(FOLDER_MODE)  # The umask can hide bits in mkdir
    return path


def write_file(config: StorageConfig, data: bytes) -> str:
    """Write a new file. Return its path relative to the root.

    The data goes to a temporary file first. After `fsync` the temporary file
    replaces the final name in one step, so a reader never sees half a file.
    """
    assert isinstance(data, bytes) and data, "data must be non-empty bytes"
    if free_space(config) - len(data) < config.min_free_bytes:
        raise StorageFullError("The disk has too little free space for a new image")

    now: datetime = datetime.now(UTC)
    folder: str = f"{now:%Y/%m/%d}"
    temp_path: Path | None = None
    try:
        directory: Path = _make_folders(config, folder)
        for _ in range(NAME_ATTEMPTS):
            name: str = f"{uuid.uuid4().hex}.jpg"
            final_path: Path = directory / name
            if not final_path.exists():
                break
        else:
            raise StorageIOError("Cannot find a free file name")
        temp_path = final_path.with_name(name + TEMP_SUFFIX)
        descriptor: int = os.open(
            temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE
        )
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, final_path)
        temp_path = None
    except OSError as error:
        logger.error("Cannot write a file: %s: %s", type(error).__name__, error)
        if error.errno == errno.ENOSPC:
            raise StorageFullError("The disk is full") from error
        raise StorageIOError("Cannot write the file") from error
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    relative: str = f"{folder}/{name}"
    logger.info("Saved %s (%d bytes)", relative, len(data))
    return relative


def read_file(config: StorageConfig, relative_path: str) -> bytes:
    """Return the bytes of a stored file."""
    path: Path = resolve_inside_root(config, relative_path)
    try:
        return path.read_bytes()
    except FileNotFoundError as error:
        raise StorageNotFoundError(f"No stored file at {relative_path}") from error
    except OSError as error:
        logger.error("Cannot read %s: %s", relative_path, error)
        raise StorageIOError(f"Cannot read {relative_path}") from error


def _remove_empty_folders(config: StorageConfig, file_path: Path) -> None:
    """Remove the day, month and year folders of a deleted file if they are empty."""
    folder: Path = file_path.parent
    root: Path = config.root.resolve()
    while folder != root and folder.is_relative_to(root):
        try:
            folder.rmdir()  # Fails if the folder is not empty
        except OSError:
            return
        folder = folder.parent


def delete_files(config: StorageConfig, relative_paths: list[str]) -> DeleteReport:
    """Delete files. All paths are checked first: one bad path stops the whole call."""
    assert isinstance(relative_paths, list), "relative_paths must be a list"
    resolved: list[tuple[str, Path]] = [
        (p, resolve_inside_root(config, p)) for p in relative_paths
    ]
    deleted: list[str] = []
    missing: list[str] = []
    failed: list[str] = []
    for relative, path in resolved:
        try:
            path.unlink()
            deleted.append(relative)
            _remove_empty_folders(config, path)
        except FileNotFoundError:
            missing.append(relative)
        except OSError as error:
            logger.warning("Cannot delete %s: %s", relative, type(error).__name__)
            failed.append(relative)
    logger.info(
        "Delete: %d deleted, %d missing, %d failed",
        len(deleted),
        len(missing),
        len(failed),
    )
    return DeleteReport(deleted, missing, failed)


def list_paths(config: StorageConfig) -> list[str]:
    """Return the relative paths of all stored files, sorted. Other files are ignored."""
    paths: list[str] = []
    for folder, _, names in os.walk(config.root):
        for name in names:
            relative: str = (Path(folder) / name).relative_to(config.root).as_posix()
            if PATH_PATTERN.fullmatch(relative):
                paths.append(relative)
    return sorted(paths)


def find_orphans(
    config: StorageConfig,
    known_paths: Iterable[str],
    min_age_seconds: int | None = None,
) -> list[str]:
    """Return stored files that are not in `known_paths` and are old enough.

    A file that is too new may be an upload that has no database row yet.
    """
    age_limit: int = (
        config.orphan_grace_seconds if min_age_seconds is None else min_age_seconds
    )
    assert age_limit >= 0, "min_age_seconds must not be negative"
    known: set[str] = set(known_paths)
    now: float = time.time()
    orphans: list[str] = []
    for relative in list_paths(config):
        if relative in known:
            continue
        try:
            age: float = now - (config.root / relative).stat().st_mtime
        except OSError:
            continue  # The file went away while we looked
        if age >= age_limit:
            orphans.append(relative)
    return orphans


def find_missing(config: StorageConfig, known_paths: Iterable[str]) -> list[str]:
    """Return the paths in `known_paths` that have no file on disk."""
    missing: list[str] = []
    for relative in sorted(set(known_paths)):
        if (
            not PATH_PATTERN.fullmatch(relative)
            or not (config.root / relative).is_file()
        ):
            missing.append(relative)
    return missing


def remove_stale_temp_files(config: StorageConfig) -> int:
    """Delete temporary files that a stopped write left behind. Return how many."""
    now: float = time.time()
    removed: int = 0
    for folder, _, names in os.walk(config.root):
        for name in names:
            path: Path = Path(folder) / name
            if not name.endswith(TEMP_SUFFIX):
                continue
            try:
                if now - path.stat().st_mtime >= config.orphan_grace_seconds:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
    if removed:
        logger.warning("Removed %d stale temporary file(s)", removed)
    return removed
