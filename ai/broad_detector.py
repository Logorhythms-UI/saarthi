"""Broad object detector using YOLO.

Produces high-recall object proposals for every uploaded image.
This is the first stage of the ingestion pipeline: detect as many
potentially useful objects as possible. Recall is favoured over precision.

These proposals feed into embedding generation — NOT final verification.
Final open-vocabulary verification is performed by Grounding DINO Tiny.
"""

import logging
import time
from dataclasses import dataclass, field

from PIL import Image
from ultralytics import YOLO

logger: logging.Logger = logging.getLogger(__name__)

# YOLO nano — small and fast; weights are auto-downloaded by ultralytics on first use.
_MODEL_NAME: str = "yolov8n.pt"

# Low confidence threshold to maximise recall at the proposal stage.
# Missing an object here means it can never enter the retrieval index.
_CONFIDENCE_THRESHOLD: float = 0.25


@dataclass
class Detection:
    """A single object detection from the broad detector.

    Attributes:
        label: Human-readable class name from the detector.
        confidence: Detector confidence score in [0, 1].
        bounding_box: [x1, y1, x2, y2] in absolute pixel coordinates.
    """

    label: str
    confidence: float
    bounding_box: list[float] = field(default_factory=list)


class BroadDetector:
    """High-recall YOLO-based object proposal generator.

    The model is loaded once and remains in memory for the application lifetime.
    Its purpose is to enumerate object instances in an image for downstream
    embedding and indexing. It is NOT a semantic verifier.
    """

    def __init__(self) -> None:
        logger.info("Loading BroadDetector (%s)", _MODEL_NAME)
        start: float = time.perf_counter()
        self._model: YOLO = YOLO(_MODEL_NAME)
        duration: float = time.perf_counter() - start
        logger.info("BroadDetector loaded in %.2fs", duration)

    def detect(self, image: Image.Image) -> list[Detection]:
        """Run broad object detection on a PIL image.

        Returns all detections above the confidence threshold. An empty list
        is a normal outcome for images with no recognisable objects at the
        chosen threshold — it is not treated as an error.

        Args:
            image: PIL image to analyse (any mode; converted to RGB internally).

        Returns:
            List of Detection objects, possibly empty.
        """
        assert isinstance(image, Image.Image), "image must be a PIL Image"

        rgb_image: Image.Image = image.convert("RGB")
        start: float = time.perf_counter()

        # verbose=False suppresses ultralytics per-frame console output
        results = self._model(rgb_image, conf=_CONFIDENCE_THRESHOLD, verbose=False)
        duration: float = time.perf_counter() - start

        detections: list[Detection] = []
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            logger.info("BroadDetector: no objects detected (%.2fs)", duration)
            return detections

        class_names: dict[int, str] = result.names

        for i in range(len(boxes)):
            box = boxes[i]
            label: str = class_names[int(box.cls.item())]
            confidence: float = float(box.conf.item())
            bounding_box: list[float] = box.xyxy[0].tolist()

            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bounding_box=bounding_box,
                )
            )

        logger.info(
            "BroadDetector: %d object(s) detected in %.2fs",
            len(detections),
            duration,
        )
        return detections
