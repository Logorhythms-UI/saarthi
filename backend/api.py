"""Application-level orchestration layer.

Coordinates AI models, database, and storage to provide
a clean interface for the UI.
"""

import logging
import time
from pathlib import Path

from peewee import IntegrityError

from ai.object_detector import ObjectDetector
from ai.query_normalizer import QueryNormalizer
from db.api import DatabaseEngine
from db.schema import ImageHistory
from storage.app import StorageEngine
from utils.result import Result

logger: logging.Logger = logging.getLogger(__name__)


class Backend:
    """Application-level orchestration layer.

    Coordinates AI models, database, and storage to provide
    a clean interface for the UI.
    """

    def __init__(self) -> None:
        self._storage: StorageEngine = StorageEngine()
        self._db: DatabaseEngine = DatabaseEngine()
        self._query_normalizer: QueryNormalizer = QueryNormalizer()
        self._object_detector: ObjectDetector = ObjectDetector()
        logger.info("Backend initialized — all subsystems ready")

    def ingest_upload(self, file_content: bytes, filename: str) -> Result:
        """Store an uploaded image and register it in the database.

        Accepts raw file bytes (from NiceGUI upload handler).
        Returns Result with the stored path on success.
        """
        assert isinstance(file_content, bytes), "file_content must be bytes"
        assert isinstance(filename, str), "filename must be a string"

        try:
            stored_path: str = self._storage.store_upload(file_content, filename)
        except (AssertionError, ValueError) as e:
            logger.error("Storage rejected upload '%s': %s", filename, e)
            return Result(err=str(e))
        except OSError as e:
            logger.error("Failed to store upload '%s': %s", filename, e)
            return Result(err=f"Failed to store image: {e}")

        try:
            self._db.add_image(stored_path)
        except IntegrityError:
            logger.info("Image already exists in database: %s", stored_path)
            return Result(result=stored_path)
        except (AssertionError, OSError) as e:
            logger.error("Database rejected image '%s': %s", stored_path, e)
            return Result(err=str(e))

        logger.info("Ingested upload: %s", filename)
        return Result(result=stored_path)

    def ingest_image(self, image_path: str) -> Result:
        """Store a single image from a file path and register in DB.

        Returns Result with the stored path on success.
        """
        assert isinstance(image_path, str), "image_path must be a string"

        try:
            stored_path: str = self._storage.upload_image(image_path)
        except AssertionError as e:
            logger.error("Invalid image path %s: %s", image_path, e)
            return Result(err=str(e))
        except (FileNotFoundError, OSError) as e:
            logger.error("Failed to store image %s: %s", image_path, e)
            return Result(err=str(e))

        try:
            self._db.add_image(stored_path)
        except IntegrityError:
            logger.info("Image already exists in database: %s", stored_path)
            return Result(result=stored_path)
        except (AssertionError, OSError) as e:
            logger.error("Database rejected image %s: %s", stored_path, e)
            return Result(err=str(e))

        logger.info("Ingested image: %s", stored_path)
        return Result(result=stored_path)

    def search(self, query: str) -> Result:
        """Execute the full search pipeline: SLM normalisation → DINO detection.

        Searches images from newest to oldest.  Stops at the first valid
        detection.  Returns a structured result with annotated image,
        detection data, and metadata.

        The result dict contains:
            annotated_image_path: str
            normalized_query: str
            original_query: str
            detection: {label, confidence, bounding_box}
            image_timestamp: str
            images_searched: int
            total_images: int
            slm_duration_s: float
            search_duration_s: float
        """
        assert isinstance(query, str), "query must be a string"
        assert len(query.strip()) > 0, "query must not be empty"

        total_start: float = time.perf_counter()
        logger.info("Search pipeline started — query: '%s'", query)

        # Phase 1: Query normalisation via SLM
        normalize_result: Result = self._query_normalizer.normalize(query)
        slm_duration: float = time.perf_counter() - total_start

        if not normalize_result.ok:
            logger.error("Query normalisation failed: %s", normalize_result.err)
            return Result(err=f"Query processing failed: {normalize_result.err}")

        normalized_query: str = normalize_result.result
        logger.info(
            "SLM normalisation completed in %.2fs: '%s' → '%s'",
            slm_duration,
            query,
            normalized_query,
        )

        # Phase 2: Get images in reverse chronological order
        images: list[ImageHistory] = self._db.get_all_images_desc()
        total_images: int = len(images)

        if total_images == 0:
            return Result(err="No images in the database. Upload images first.")

        logger.info(
            "Searching '%s' across %d images (newest → oldest)",
            normalized_query,
            total_images,
        )

        # Phase 3: Iterate and detect
        search_start: float = time.perf_counter()

        for images_searched, image_record in enumerate(images, start=1):
            image_path: str = image_record.imgpath
            logger.info(
                "Scanning image %d/%d: %s",
                images_searched,
                total_images,
                Path(image_path).name,
            )

            detect_start: float = time.perf_counter()
            detection_result: Result = self._object_detector.detect(
                image_path, normalized_query
            )
            detect_duration: float = time.perf_counter() - detect_start

            if detection_result.ok:
                detection: dict = detection_result.result
                logger.info(
                    "Match found in %s (DINO inference: %.2fs, confidence: %.2f%%)",
                    Path(image_path).name,
                    detect_duration,
                    detection["confidence"] * 100,
                )

                # Generate annotated image
                annotated_path: str = self._object_detector.draw_bounding_box(
                    image_path,
                    bounding_box=detection["bounding_box"],
                    label=detection["label"],
                    confidence=detection["confidence"],
                )

                total_duration: float = time.perf_counter() - total_start
                search_duration: float = time.perf_counter() - search_start

                logger.info(
                    "Search complete — images searched: %d/%d, "
                    "SLM: %.2fs, detection: %.2fs, total: %.2fs",
                    images_searched,
                    total_images,
                    slm_duration,
                    search_duration,
                    total_duration,
                )

                return Result(
                    result={
                        "annotated_image_path": annotated_path,
                        "normalized_query": normalized_query,
                        "original_query": query,
                        "detection": detection,
                        "image_timestamp": str(image_record.timestamp),
                        "images_searched": images_searched,
                        "total_images": total_images,
                        "slm_duration_s": round(slm_duration, 2),
                        "search_duration_s": round(
                            time.perf_counter() - search_start, 2
                        ),
                    }
                )

            logger.debug(
                "No detection in %s (%.2fs): %s",
                Path(image_path).name,
                detect_duration,
                detection_result.err,
            )

        total_duration = time.perf_counter() - total_start
        logger.info(
            "No match found — searched all %d images in %.2fs",
            total_images,
            total_duration,
        )
        return Result(
            err=f"No matching object was found for '{normalized_query}' "
            f"in any stored image."
        )

    def get_image_count(self) -> int:
        """Return the number of images in the database."""
        return self._db.get_image_count()

    def get_all_images_desc(self) -> list[ImageHistory]:
        """Return all image records ordered newest-first by sequence number."""
        return self._db.get_all_images_desc()
