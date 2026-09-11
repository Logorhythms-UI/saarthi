"""Helpers that tests of the vector features share. They use the actual db."""

import uuid

import numpy as np

from db.api import NewRegion, add_image, add_regions
from db.schema import Image, ImageStatus, Region, RegionSource

TEST_PREFIX: str = "__test__/"
MODEL: str = "test-model"
DIMENSION: int = 768  # Same as vectors.dimension in config.toml


def make_image() -> Image:
    """Make an image in the status `processing`."""
    image: Image = add_image(
        f"{TEST_PREFIX}{uuid.uuid4().hex}.jpg", uuid.uuid4().hex * 2, 640, 480
    )
    Image.update(status=ImageStatus.PROCESSING.value).where(
        Image.id == image.id
    ).execute()
    return image


def make_regions(image: Image, count: int) -> list[Region]:
    new: list[NewRegion] = [
        NewRegion(i, i, i + 10, i + 10, 0.3, RegionSource.FULL, ["a bottle"])
        for i in range(count)
    ]
    return add_regions(image.id, new)


def set_status(image: Image, status: ImageStatus) -> None:
    Image.update(status=status.value).where(Image.id == image.id).execute()


def unit_vectors(count: int, seed: int = 0) -> np.ndarray:
    rng: np.random.Generator = np.random.default_rng(seed)
    raw: np.ndarray = rng.normal(size=(count, DIMENSION)).astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)
