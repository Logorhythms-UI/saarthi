"""Finds the photo for tests with a real picture."""

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _find() -> Path | None:
    """Use SAARTHI_TEST_IMAGE, or `test_img.jpg` in the project root."""
    chosen: str | None = os.environ.get("SAARTHI_TEST_IMAGE")
    path: Path = Path(chosen) if chosen else PROJECT_ROOT / "test_img.jpg"
    return path if path.is_file() else None


TEST_IMAGE: Path | None = _find()
