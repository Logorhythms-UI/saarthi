"""Image quality checks: a blur score and a perceptual hash.

Both use numpy and Pillow only. They do not load a model.
"""

import numpy as np
from PIL import Image

HASH_SIDE: int = 8  # The hash has HASH_SIDE * HASH_SIDE bits
HASH_LENGTH: int = HASH_SIDE * HASH_SIDE // 4  # Length in hex characters


def _gray(image: Image.Image, max_side: int | None) -> np.ndarray:
    """Return a grayscale float32 copy. Shrink it so the longest side is `max_side`."""
    gray: Image.Image = image.convert("L")
    if max_side is not None and max(gray.size) > max_side:
        gray.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    return np.asarray(gray, dtype=np.float32)


def blur_score(image: Image.Image, max_side: int | None = None) -> float:
    """Return the variance of the Laplacian. A low value means a blurry image.

    Pass `max_side` for a full frame, so the value does not depend on the
    camera resolution. Do not pass it for a small crop.
    """
    assert isinstance(image, Image.Image), "image must be a PIL image"
    assert max_side is None or max_side > 0, "max_side must be positive"
    pixels: np.ndarray = _gray(image, max_side)
    if pixels.shape[0] < 3 or pixels.shape[1] < 3:
        return 0.0
    center: np.ndarray = pixels[1:-1, 1:-1]
    laplacian: np.ndarray = (
        pixels[:-2, 1:-1] + pixels[2:, 1:-1] + pixels[1:-1, :-2] + pixels[1:-1, 2:]
    ) - 4.0 * center
    return float(laplacian.var())


def perceptual_hash(image: Image.Image) -> str:
    """Return a 64-bit difference hash as 16 hex characters."""
    assert isinstance(image, Image.Image), "image must be a PIL image"
    small: Image.Image = image.convert("L").resize(
        (HASH_SIDE + 1, HASH_SIDE), Image.Resampling.LANCZOS
    )
    pixels: np.ndarray = np.asarray(small, dtype=np.int16)
    bits: np.ndarray = pixels[:, 1:] > pixels[:, :-1]
    value: int = int("".join("1" if b else "0" for b in bits.flatten()), 2)
    return f"{value:0{HASH_LENGTH}x}"


def hamming_distance(first: str, second: str) -> int:
    """Return the number of bits in which two hashes differ."""
    assert len(first) == HASH_LENGTH and len(second) == HASH_LENGTH, "bad hash length"
    return (int(first, 16) ^ int(second, 16)).bit_count()
