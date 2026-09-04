import json
import logging
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from utils.result import Result

logging.basicConfig(
    level=logging.INFO, format="AI - %(asctime)s - %(levelname)s - %(message)s"
)

# Cache file stored next to this module, persists across restarts
_CACHE_PATH: Path = Path(__file__).parent / "inference_cache.json"


def _load_cache() -> dict:
    """Load the inference cache from disk, returning an empty dict on first run."""
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    """Persist the inference cache to disk."""
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError as e:
        logging.getLogger(__name__).warning(f"Failed to save inference cache: {e}")


class AI_Engine:
    def __init__(self) -> None:
        self.logger: logging.Logger = logging.getLogger(__name__)

        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Loading Grounding DINO model on device: {self.device}")

        # Configure Model Parameters
        self._MODEL_ID: str = "IDEA-Research/grounding-dino-tiny"
        self._BOX_THRESHOLD: float = 0.35
        self._TEXT_THRESHOLD: float = 0.25

        self.processor: AutoProcessor = AutoProcessor.from_pretrained(self._MODEL_ID)
        self.model: AutoModelForZeroShotObjectDetection = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self._MODEL_ID).to(
                self.device
            )
        )
        self.logger.info("Grounding DINO model loaded and ready")

        # Load persisted inference cache so results survive app restarts
        self._cache: dict = _load_cache()
        self.logger.info(f"Inference cache loaded: {len(self._cache)} entries")

    def detect(self, image_path: str, query: str) -> Result:
        """Run zero-shot object detection on a single image.

        The query is automatically lowercased and period-suffixed as required
        by Grounding DINO's tokenisation.

        Results are cached to disk so that repeated searches (or app restarts)
        never re-run the model on the same image+query pair.

        Returns Result with detection dict on success (keys: boxes, scores,
        labels), or Result with err if nothing was detected.
        """
        assert isinstance(image_path, str), "image_path must be a string"
        assert isinstance(query, str), "query must be a string"
        assert Path(image_path).exists(), f"Image {image_path} does not exist"

        # Grounding DINO requires lowercase prompt ending with a period
        prompt: str = query.lower().strip()
        if not prompt.endswith("."):
            prompt = prompt + "."

        # --- Cache lookup ---
        cache_key: str = f"{image_path}|||{prompt}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached is None:
                # Previously found no detections — skip inference
                self.logger.info(
                    f"Cache hit (no detection) for '{query}' "
                    f"in {Path(image_path).name}"
                )
                return Result(err=f"No '{query}' detected in {Path(image_path).name}")
            self.logger.info(f"Cache hit for '{query}' in {Path(image_path).name}")
            return Result(result=cached)

        try:
            image: Image.Image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            results: list[dict] = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self._BOX_THRESHOLD,
                text_threshold=self._TEXT_THRESHOLD,
                target_sizes=[image.size[::-1]],
            )

            detections: dict = results[0]
            boxes = detections["boxes"]

            if len(boxes) == 0:
                # Cache the negative result so we skip this image next time
                self._cache[cache_key] = None
                _save_cache(self._cache)
                return Result(err=f"No '{query}' detected in {Path(image_path).name}")

            detection_data: dict = {
                "boxes": boxes.cpu().tolist(),
                "scores": detections["scores"].cpu().tolist(),
                "labels": detections["labels"],
            }
            self.logger.info(
                f"Detected {len(boxes)} object(s) for '{query}' "
                f"in {Path(image_path).name}"
            )

            # Cache and persist the positive result
            self._cache[cache_key] = detection_data
            _save_cache(self._cache)

            return Result(result=detection_data)

        except UnidentifiedImageError as e:
            self.logger.error(f"Unable to open image {image_path}: {e}")
            return Result(err=f"Unable to open image: {e}")
        except FileNotFoundError as e:
            self.logger.error(f"Image not found {image_path}: {e}")
            return Result(err=str(e))
        except RuntimeError as e:
            msg = str(e)
            if "out of memory" in msg.lower():
                self.logger.error(f"CUDA OOM during detection on {image_path}: {e}")
                return Result(
                    err="Out of memory during detection. Try using CPU or a smaller image."
                )
            self.logger.error(f"Runtime error during detection on {image_path}: {e}")
            return Result(err=str(e))
        except Exception:
            # Re-raise unexpected exceptions so they are not silently swallowed
            raise

    def clear_cache(self) -> None:
        """Wipe the in-memory inference cache and delete the on-disk cache file.

        Call this after deleting or re-labelling stored images so that
        subsequent searches always re-run the model on fresh data.
        """
        self._cache = {}
        if _CACHE_PATH.exists():
            try:
                _CACHE_PATH.unlink()
                self.logger.info("Inference cache cleared from disk")
            except OSError as e:
                self.logger.warning(f"Failed to delete cache file: {e}")
        else:
            self.logger.info("No on-disk cache to delete")

    def draw_bounding_boxes(
        self,
        image_path: str,
        boxes: list[list[float]],
        labels: list[str],
        scores: list[float],
    ) -> str:
        """Draw bounding boxes on a copy of the image and save it.

        Returns the absolute path of the annotated image saved alongside
        the original with an '_annotated' suffix.
        """
        assert isinstance(image_path, str), "image_path must be a string"
        assert Path(image_path).exists(), f"Image {image_path} does not exist"
        assert (
            len(boxes) == len(labels) == len(scores)
        ), "boxes, labels, and scores must have equal length"

        image: Image.Image = Image.open(image_path).convert("RGB")
        draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)

        # Attempt to use a readable font size, fall back to default
        try:
            font_size: int = max(16, min(image.width, image.height) // 30)
            font: ImageFont.FreeTypeFont = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
            )
        except OSError:
            self.logger.info("TrueType font not found, using default font")
            font = ImageFont.load_default()

        for box, label, score in zip(boxes, labels, scores):
            x1: float = box[0]
            y1: float = box[1]
            x2: float = box[2]
            y2: float = box[3]

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="#FF3B30", width=3)

            # Draw label background and text
            text: str = f"{label} {score:.0%}"
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

        # Save annotated image alongside original
        src: Path = Path(image_path)
        annotated_path: Path = src.parent / f"{src.stem}_annotated{src.suffix}"
        image.save(str(annotated_path))
        self.logger.info(f"Annotated image saved to {annotated_path}")
        return str(annotated_path)
