"""Tests of the file helpers. They use a temporary root, because they make links and old files."""

import dataclasses
import errno
import os
import re
import shutil
import time
import uuid
from collections import namedtuple
from pathlib import Path

import pytest

from storage import files
from storage.errors import (
    StorageFullError,
    StorageIOError,
    StorageNotFoundError,
    StoragePathError,
)
from storage.settings import StorageConfig, load_config

DATA: bytes = b"\xff\xd8 not a real jpeg but it is only bytes \xff\xd9"


@pytest.fixture()
def config(tmp_path: Path) -> StorageConfig:
    root: Path = tmp_path / "root"
    configuration: StorageConfig = dataclasses.replace(
        load_config(), root=root, min_free_bytes=0
    )
    files.ensure_root(configuration)
    return configuration


def stored_name() -> str:
    return f"2026/10/25/{uuid.uuid4().hex}.jpg"


def place(config: StorageConfig, relative: str, age_seconds: int = 0) -> Path:
    """Put a file at a path, as an older upload."""
    path: Path = config.root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(DATA)
    old: float = time.time() - age_seconds
    os.utime(path, (old, old))
    return path


def test_write_makes_a_private_file_with_a_random_name(config: StorageConfig) -> None:
    first: str = files.write_file(config, DATA)
    second: str = files.write_file(config, DATA)
    assert (
        re.fullmatch(r"\d{4}/\d{2}/\d{2}/[0-9a-f]{32}\.jpg", first) and first != second
    )
    path: Path = config.root / first
    assert path.read_bytes() == DATA
    assert (path.stat().st_mode & 0o777) == 0o600
    for folder in (path.parent, path.parent.parent, path.parent.parent.parent):
        assert (folder.stat().st_mode & 0o777) == 0o700
    assert not list(config.root.rglob("*.tmp"))


def test_read_returns_the_bytes_and_a_missing_file_is_reported(
    config: StorageConfig,
) -> None:
    relative: str = files.write_file(config, DATA)
    assert files.read_file(config, relative) == DATA
    with pytest.raises(StorageNotFoundError):
        files.read_file(config, stored_name())


BAD_PATHS: list[str] = [
    "",
    "../x.jpg",
    "/etc/passwd",
    "2026/10/25/../../../../etc/passwd",
    "2026/10/25/" + "a" * 32 + ".png",
    "2026/10/25/" + "A" * 32 + ".jpg",  # Not lowercase hex
    "2026/10/25/" + "a" * 31 + ".jpg",
    "2026\\10\\25\\" + "a" * 32 + ".jpg",
    "2026/10/25/" + "a" * 32 + ".jpg\n",
    "2026/10/25/" + "a" * 32 + ".jpg\x00",
    "/2026/10/25/" + "a" * 32 + ".jpg",
]


@pytest.mark.parametrize("path", BAD_PATHS)
def test_bad_paths_are_refused_by_every_helper(
    config: StorageConfig, path: str
) -> None:
    with pytest.raises(StoragePathError):
        files.read_file(config, path)
    with pytest.raises(StoragePathError):
        files.delete_files(config, [path])


def test_a_link_that_leads_out_of_the_root_is_refused(
    config: StorageConfig, tmp_path: Path
) -> None:
    outside: Path = tmp_path / "outside"
    outside.mkdir()
    name: str = uuid.uuid4().hex + ".jpg"
    (outside / name).write_bytes(b"secret")
    (config.root / "2026" / "10").mkdir(parents=True)
    (config.root / "2026" / "10" / "25").symlink_to(outside, target_is_directory=True)
    with pytest.raises(StoragePathError):
        files.read_file(config, f"2026/10/25/{name}")
    with pytest.raises(StoragePathError):
        files.delete_files(config, [f"2026/10/25/{name}"])
    assert (outside / name).exists()


def test_delete_reports_each_result_and_removes_empty_folders(
    config: StorageConfig,
) -> None:
    kept: str = stored_name()
    gone: str = f"2025/01/02/{uuid.uuid4().hex}.jpg"
    place(config, kept)
    place(config, gone)
    absent: str = f"2025/01/02/{uuid.uuid4().hex}.jpg"
    report = files.delete_files(config, [gone, absent])
    assert (
        report.deleted == [gone] and report.missing == [absent] and report.failed == []
    )
    assert not (
        config.root / "2025"
    ).exists()  # The empty year, month and day folders are gone
    assert (config.root / kept).exists() and config.root.exists()


def test_a_bad_path_stops_the_whole_delete_before_anything_is_deleted(
    config: StorageConfig,
) -> None:
    good: str = stored_name()
    place(config, good)
    with pytest.raises(StoragePathError):
        files.delete_files(config, [good, "../x.jpg"])
    assert (config.root / good).exists()


def test_a_refused_delete_is_reported_as_failed(
    config: StorageConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative: str = stored_name()
    place(config, relative)

    def refuse(self: Path, *args: object, **kwargs: object) -> None:
        raise PermissionError(errno.EACCES, "no")

    monkeypatch.setattr(Path, "unlink", refuse)
    report = files.delete_files(config, [relative])
    assert report.failed == [relative] and report.deleted == []


def test_list_paths_ignores_temporary_and_foreign_files(config: StorageConfig) -> None:
    first, second = sorted([stored_name(), stored_name()])
    place(config, second)
    place(config, first)
    (config.root / "2026/10/25" / (uuid.uuid4().hex + ".jpg.tmp")).write_bytes(b"x")
    (config.root / "notes.txt").write_text("x")
    assert files.list_paths(config) == [first, second]


def test_find_orphans_uses_the_grace_period(config: StorageConfig) -> None:
    known: str = stored_name()
    old_orphan: str = stored_name()
    new_orphan: str = stored_name()
    place(config, known, age_seconds=7200)
    place(config, old_orphan, age_seconds=7200)
    place(config, new_orphan, age_seconds=10)
    assert files.find_orphans(config, [known]) == [old_orphan]  # The default is 3600 s
    assert set(files.find_orphans(config, [known], min_age_seconds=0)) == {
        old_orphan,
        new_orphan,
    }
    assert files.find_orphans(config, [known, old_orphan, new_orphan], 0) == []


def test_find_missing_lists_database_paths_without_a_file(
    config: StorageConfig,
) -> None:
    present: str = stored_name()
    place(config, present)
    absent: str = stored_name()
    assert files.find_missing(
        config, [present, absent, "not-a-path", absent]
    ) == sorted([absent, "not-a-path"])


def test_a_full_disk_is_refused_and_nothing_is_written(
    config: StorageConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(100, 90, 10))
    limited: StorageConfig = dataclasses.replace(config, min_free_bytes=5)
    with pytest.raises(StorageFullError):
        files.write_file(limited, DATA)  # 10 bytes free, minus the data, is below 5
    assert not list(config.root.rglob("*"))


def test_a_disk_that_fills_up_during_the_write_is_reported_and_leaves_no_temporary_file(
    config: StorageConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(descriptor: int) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(StorageFullError):
        files.write_file(config, DATA)
    assert not list(config.root.rglob("*.tmp")) and files.list_paths(config) == []


def test_a_failed_rename_leaves_no_temporary_file(
    config: StorageConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(source: object, target: object) -> None:
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(StorageIOError):
        files.write_file(config, DATA)
    assert not list(config.root.rglob("*.tmp")) and files.list_paths(config) == []


def test_stale_temporary_files_are_removed_and_fresh_ones_stay(
    config: StorageConfig,
) -> None:
    old: Path = place(config, stored_name() + ".tmp", age_seconds=7200)
    fresh: Path = place(config, stored_name() + ".tmp", age_seconds=10)
    assert files.remove_stale_temp_files(config) == 1
    assert not old.exists() and fresh.exists()


def test_ensure_root_creates_a_private_folder(tmp_path: Path) -> None:
    root: Path = tmp_path / "a" / "b"
    files.ensure_root(dataclasses.replace(load_config(), root=root))
    assert root.is_dir() and (root.stat().st_mode & 0o777) == 0o700


def test_ensure_root_refuses_a_file_and_a_folder_that_cannot_be_written(
    tmp_path: Path,
) -> None:
    a_file: Path = tmp_path / "file"
    a_file.write_text("x")
    with pytest.raises(StorageIOError):
        files.ensure_root(dataclasses.replace(load_config(), root=a_file))
    if os.geteuid() == 0:
        pytest.skip("root can write to any folder")
    locked: Path = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    try:
        with pytest.raises(StorageIOError):
            files.ensure_root(dataclasses.replace(load_config(), root=locked))
    finally:
        locked.chmod(0o700)
