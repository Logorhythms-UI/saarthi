"""Object detector using Grounding DINO Tiny.

Provides zero-shot object detection on images using the
IDEA-Research/grounding-dino-tiny model from Hugging Face.
"""

import logging
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from utils.result import Result

logger: logging.Logger = logging.getLogger(__name__)

_MODEL_ID: str = "IDEA-Research/grounding-dino-tiny"
_BOX_THRESHOLD: float = 0.65
_TEXT_THRESHOLD: float = 0.90


class ObjectDetector:
    """Zero-shot object detector using Grounding DINO Tiny.

    The model is loaded once and remains in memory for the
    application lifetime.
    """

    def __init__(self) -> None:
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Grounding DINO on device: %s", self._device)

        start: float = time.perf_counter()
        self._processor: AutoProcessor = AutoProcessor.from_pretrained(_MODEL_ID)
        self._model: AutoModelForZeroShotObjectDetection = (
            AutoModelForZeroShotObjectDetection.from_pretrained(_MODEL_ID).to(
                self._device
            )
        )
        duration: float = time.perf_counter() - start
        logger.info("Grounding DINO loaded in %.2fs on %s", duration, self._device)

    def detect(self, image_path: str, query: str) -> Result:
        """Run zero-shot object detection on a single image.

        Returns Result with a detection dict on success containing:
            label: str
            confidence: float
            bounding_box: [x1, y1, x2, y2]

        Only the highest-confidence detection is returned.
        Returns Result with err if nothing was detected or on failure.
        """
        assert isinstance(image_path, str), "image_path must be a string"
        assert isinstance(query, str), "query must be a string"
        assert Path(image_path).exists(), f"Image {image_path} does not exist"
        assert len(query.strip()) > 0, "query must not be empty"

        # Grounding DINO requires lowercase prompt ending with a period
        prompt: str = query.lower().strip()
        if not prompt.endswith("."):
            prompt = prompt + "."

        try:
            image: Image.Image = Image.open(image_path).convert("RGB")
        except UnidentifiedImageError as e:
            logger.error("Unable to open image %s: %s", image_path, e)
            return Result(err=f"Unable to process image: {e}")
        except FileNotFoundError as e:
            logger.error("Image not found %s: %s", image_path, e)
            return Result(err=f"Unable to process image: {e}")

        try:
            inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(
                self._device
            )

            with torch.no_grad():
                outputs = self._model(**inputs)

            results: list[dict] = (
                self._processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=_BOX_THRESHOLD,
                    text_threshold=_TEXT_THRESHOLD,
                    target_sizes=[image.size[::-1]],
                )
            )
        except RuntimeError as e:
            msg: str = str(e)
            if "out of memory" in msg.lower():
                logger.error("CUDA OOM during detection on %s: %s", image_path, e)
                return Result(err="Object detection failed: out of memory")
            logger.error("Runtime error during detection on %s: %s", image_path, e)
            return Result(err=f"Object detection failed: {e}")

        detections: dict = results[0]
        boxes = detections["boxes"]

        if len(boxes) == 0:
            return Result(err=f"No '{query}' detected in {Path(image_path).name}")

        # Select highest-confidence detection
        scores: list[float] = detections["scores"].cpu().tolist()
        best_idx: int = scores.index(max(scores))
        box: list[float] = boxes[best_idx].cpu().tolist()

        # Validate bounding box coordinates
        validated_box: list[float] = _validate_bounding_box(
            box, image.width, image.height
        )

        detection_data: dict = {
            "label": detections["labels"][best_idx],
            "confidence": scores[best_idx],
            "bounding_box": validated_box,
        }

        logger.info(
            "Detected '%s' (confidence: %.2f%%) in %s",
            detection_data["label"],
            detection_data["confidence"] * 100,
            Path(image_path).name,
        )
        return Result(result=detection_data)

    def draw_bounding_box(
        self,
        image_path: str,
        bounding_box: list[float],
        label: str,
        confidence: float,
    ) -> str:
        """Draw a bounding box on a copy of the image.

        Saves the annotated image to a temporary file to preserve
        the original.  Returns the absolute path of the temp file.
        """
        assert isinstance(image_path, str), "image_path must be a string"
        assert Path(image_path).exists(), f"Image {image_path} does not exist"
        assert len(bounding_box) == 4, "bounding_box must have 4 coordinates"

        image: Image.Image = Image.open(image_path).convert("RGB")
        draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)

        try:
            font_size: int = max(16, min(image.width, image.height) // 30)
            font: ImageFont.FreeTypeFont = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
            )
        except OSError:
            logger.info("TrueType font not found, using default font")
            font = ImageFont.load_default()

        x1, y1, x2, y2 = bounding_box
        draw.rectangle([x1, y1, x2, y2], outline="#FF3B30", width=3)

        text: str = f"{label} {confidence:.0%}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(
            [
                text_bbox[0] - 2,
                text_bbox[1] - 2,
                text_bbox[2] + 2,
                text_bbox[3] + 2,
            ],
            fill="#FF3B30",
        )
        draw.text((x1, y1), text, fill="white", font=font)

        # Save to temporary file — original image stays untouched
        suffix: str = Path(image_path).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="saarthi_annotated_"
        ) as tmp:
            image.save(tmp.name)
            annotated_path: str = tmp.name

        logger.info("Annotated image saved to %s", annotated_path)
        return annotated_path


def _validate_bounding_box(
    box: list[float], img_width: int, img_height: int
) -> list[float]:
    """Validate and clamp bounding box coordinates to image bounds.

    Ensures x1 <= x2, y1 <= y2, and all coordinates are within
    the image dimensions.
    """
    x1: float = max(0.0, min(box[0], box[2]))
    y1: float = max(0.0, min(box[1], box[3]))
    x2: float = min(float(img_width), max(box[0], box[2]))
    y2: float = min(float(img_height), max(box[1], box[3]))
    return [x1, y1, x2, y2]