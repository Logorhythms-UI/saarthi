"""Tests with the real models. They download the weights on the first run.

Run them with: uv run pytest -m slow
Set SAARTHI_TEST_IMAGE to the path of a photo to also test detection on a real picture.
"""

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ai.detector import CandidateDetector
from ai.embedder import RegionEmbedder
from ai.postprocess import select_candidates
from ai.settings import AiConfig, load_config
from ai.types import Box, RawDetection

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def config() -> AiConfig:
    return load_config()


@pytest.fixture(scope="module")
def embedder(config: AiConfig) -> RegionEmbedder:
    return RegionEmbedder(config)


@pytest.fixture(scope="module")
def detector(config: AiConfig) -> CandidateDetector:
    return CandidateDetector(config)


def synthetic_image(size: tuple[int, int] = (640, 480)) -> Image.Image:
    rng: np.random.Generator = np.random.default_rng(0)
    image: Image.Image = Image.fromarray(
        rng.integers(80, 160, (size[1], size[0], 3), dtype=np.uint8)
    )
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 300, 300), fill=(200, 30, 30))
    draw.ellipse((400, 200, 520, 320), fill=(30, 30, 200))
    return image


def test_embedder_output_shape_and_values(embedder: RegionEmbedder) -> None:
    boxes: list[Box] = [
        Box(100, 100, 300, 300),  # Large: pooled from the frame
        Box(400, 200, 520, 320),  # Large
        Box(10, 10, 50, 50),  # Small: embedded from a crop
    ]
    vectors: np.ndarray = embedder.embed(synthetic_image(), boxes)
    assert embedder.dimension == 384
    assert vectors.shape == (3, 384) and vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert embedder.name.startswith("dinov2-small@")


def test_embedder_is_repeatable_and_tells_regions_apart(
    embedder: RegionEmbedder,
) -> None:
    image: Image.Image = synthetic_image()
    boxes: list[Box] = [Box(100, 100, 300, 300), Box(400, 200, 520, 320)]
    first: np.ndarray = embedder.embed(image, boxes)
    second: np.ndarray = embedder.embed(image, boxes)
    assert np.allclose(first, second, atol=1e-5)
    assert (
        float(first[0] @ first[1]) < 0.99
    )  # Different regions are not the same vector


def test_embedder_without_boxes(embedder: RegionEmbedder) -> None:
    assert embedder.embed(synthetic_image(), []).shape == (0, 384)


def test_embedder_refuses_a_box_outside_the_image(embedder: RegionEmbedder) -> None:
    with pytest.raises(AssertionError):
        embedder.embed(synthetic_image(), [Box(600, 400, 700, 500)])


def test_detector_runs_all_three_passes(
    detector: CandidateDetector, config: AiConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiled: AiConfig = dataclasses.replace(
        config, detection=dataclasses.replace(config.detection, tiling_enabled=True)
    )
    monkeypatch.setattr(detector, "_config", tiled)
    image: Image.Image = synthetic_image()
    detections: list[RawDetection] = detector.detect(
        image,
        config.detection.default_phrases,
        config.detection.small_phrases,
        include_generic=True,
    )
    for detection in detections:
        box: Box = detection.box
        assert 0 <= box.x0 < box.x1 <= image.width
        assert 0 <= box.y0 < box.y1 <= image.height
        assert detection.source in ("full", "tile", "generic")


def test_token_count(detector: CandidateDetector) -> None:
    assert detector.count_tokens("a pair of glasses") >= 4


@pytest.mark.skipif(
    not os.environ.get("SAARTHI_TEST_IMAGE"), reason="SAARTHI_TEST_IMAGE is not set"
)
def test_real_photo_end_to_end(
    detector: CandidateDetector, embedder: RegionEmbedder, config: AiConfig
) -> None:
    image: Image.Image = Image.open(Path(os.environ["SAARTHI_TEST_IMAGE"])).convert(
        "RGB"
    )
    raw: list[RawDetection] = detector.detect(image, config.detection.default_phrases)
    candidates = select_candidates(
        raw,
        image.width,
        image.height,
        config.detection.default_phrases,
        config.detection,
    )
    vectors: np.ndarray = embedder.embed(image, [c.box for c in candidates])
    assert vectors.shape == (len(candidates), 384)
