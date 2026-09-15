import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ai.quality import HASH_LENGTH, blur_score, hamming_distance, perceptual_hash


def scene(size: tuple[int, int] = (640, 480), seed: int = 0) -> Image.Image:
    """Make a repeatable picture with edges and some texture."""
    rng: np.random.Generator = np.random.default_rng(seed)
    pixels: np.ndarray = rng.integers(
        60, 200, size=(size[1], size[0], 3), dtype=np.uint8
    )
    image: Image.Image = Image.fromarray(pixels)
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)
    for _ in range(12):
        x, y = int(rng.integers(0, size[0] - 80)), int(rng.integers(0, size[1] - 80))
        color: tuple[int, int, int] = tuple(int(c) for c in rng.integers(0, 255, 3))
        draw.rectangle((x, y, x + 70, y + 60), fill=color)
    return image


def test_flat_image_has_no_blur_score() -> None:
    assert blur_score(Image.new("RGB", (200, 200), (128, 128, 128))) == 0.0


def test_blurred_image_scores_lower_than_sharp() -> None:
    sharp: Image.Image = scene()
    blurred: Image.Image = sharp.filter(ImageFilter.GaussianBlur(6))
    assert blur_score(blurred) < blur_score(sharp) * 0.2


def test_blur_score_with_max_side_keeps_the_order() -> None:
    sharp: Image.Image = scene((2000, 1500))
    blurred: Image.Image = sharp.filter(ImageFilter.GaussianBlur(12))
    assert blur_score(blurred, max_side=512) < blur_score(sharp, max_side=512)


def test_blur_score_of_a_tiny_image_is_zero() -> None:
    assert blur_score(Image.new("RGB", (2, 2))) == 0.0


def test_phash_has_16_hex_characters_and_is_stable() -> None:
    image: Image.Image = scene()
    first: str = perceptual_hash(image)
    assert len(first) == HASH_LENGTH == 16
    int(first, 16)
    assert perceptual_hash(image.copy()) == first


def test_phash_survives_small_changes_and_separates_different_images() -> None:
    image: Image.Image = scene()
    noise: np.ndarray = np.random.default_rng(1).integers(-3, 4, (480, 640, 3))
    noisy: Image.Image = Image.fromarray(
        np.clip(np.asarray(image, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    )
    resized: Image.Image = image.resize((320, 240))
    base: str = perceptual_hash(image)
    assert hamming_distance(base, perceptual_hash(noisy)) <= 4
    assert hamming_distance(base, perceptual_hash(resized)) <= 4
    assert hamming_distance(base, perceptual_hash(scene(seed=7))) > 10


def test_hamming_distance() -> None:
    assert hamming_distance("0" * 16, "0" * 16) == 0
    assert hamming_distance("0" * 16, "f" * 16) == 64
    assert hamming_distance("0" * 15 + "1", "0" * 15 + "3") == 1
