import logging
import zipfile

# Import specific DB exception used by peewee for constraint errors
from peewee import IntegrityError

from ai.app import AI_Engine
from db.api import db_engine
from db.schema import ImageHistory
from storage.app import storage_engine
from utils.result import Result

logging.basicConfig(
    level=logging.INFO, format="BACKEND - %(asctime)s - %(levelname)s - %(message)s"
)


class backend_api:
    def __init__(self) -> None:
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.storage_api: storage_engine = storage_engine()
        self.db_api: db_engine = db_engine()
        self.ai_api: AI_Engine = AI_Engine()
        self.logger.info("Backend API Started and ready to go")

    def ingest_image(self, image_path: str) -> Result:
        """Store a single image and register it in the database.

        Returns Result with the stored path on success.
        """
        assert isinstance(image_path, str), "image_path must be a string"

        try:
            stored_path: str = self.storage_api.upload_image(image_path)
        except AssertionError as e:
            self.logger.error(f"Invalid image path {image_path}: {e}")
            return Result(err=str(e))
        except (FileNotFoundError, OSError) as e:
            self.logger.error(f"Failed to store image {image_path}: {e}")
            return Result(err=str(e))

        try:
            self.db_api.add_image(stored_path)
        except AssertionError as e:
            # Should not happen because storage confirmed file exists, but handle defensively
            self.logger.error(f"Database rejected image {stored_path}: {e}")
            return Result(err=str(e))
        except IntegrityError:
            # Image already registered; treat as success but inform caller
            self.logger.info(f"Image already exists in DB: {stored_path}")
            return Result(result=stored_path)

        self.logger.info(f"Ingested image: {stored_path}")
        return Result(result=stored_path)

    def ingest_zip(self, zip_path: str) -> Result:
        """Extract images from a zip and register each in the database.

        Returns Result with the count of successfully ingested images.
        """
        assert isinstance(zip_path, str), "zip_path must be a string"

        try:
            extracted_paths: list[str] = self.storage_api.extract_zip(zip_path)
        except AssertionError as e:
            self.logger.error(f"Invalid zip path {zip_path}: {e}")
            return Result(err=str(e))
        except zipfile.BadZipFile as e:
            self.logger.error(f"Bad zip file {zip_path}: {e}")
            return Result(err=str(e))
        except (FileNotFoundError, OSError) as e:
            self.logger.error(f"Failed to extract zip {zip_path}: {e}")
            return Result(err=str(e))

        ingested_count: int = 0
        for img_path in extracted_paths:
            try:
                # Image is already in uploads after extraction, register in DB
                self.db_api.add_image(img_path)
                ingested_count += 1
            except IntegrityError:
                # Skip duplicates; continue ingesting the rest
                self.logger.info(f"Image already registered, skipping: {img_path}")
                continue
            except AssertionError as e:
                self.logger.error(f"Invalid image during ingest {img_path}: {e}")
                continue
            except OSError as e:
                self.logger.error(f"OS error while registering {img_path}: {e}")
                continue

        self.logger.info(f"Ingested {ingested_count} images from zip {zip_path}")
        return Result(result=ingested_count)

    def search(self, query: str) -> Result:
        """Search for an object across all stored images in reverse chronological order.

        Iterates from the most recent image to the oldest. On the first
        detection match, draws bounding boxes on the image and returns the
        annotated image path via Result.

        Returns Result with err if no match is found across all images.
        """
        assert isinstance(query, str), "query must be a string"
        assert len(query.strip()) > 0, "query must not be empty"

        images: list[ImageHistory] = self.db_api.get_all_images_desc()

        if len(images) == 0:
            return Result(err="No images in the database. Upload images first.")

        self.logger.info(
            f"Searching for '{query}' across {len(images)} images "
            f"(newest to oldest)"
        )

        for image_record in images:
            img_path: str = image_record.imgpath
            self.logger.info(f"Scanning: {img_path}")

            detection_result: Result = self.ai_api.detect(img_path, query)

            if detection_result.ok:
                data = detection_result.result
                annotated_path: str = self.ai_api.draw_bounding_boxes(
                    img_path,
                    boxes=data["boxes"],
                    labels=data["labels"],
                    scores=data["scores"],
                )
                self.logger.info(f"Match found for '{query}' in {img_path}")
                return Result(result=annotated_path)

        self.logger.info(f"No match for '{query}' in any image")
        return Result(err=f"'{query}' was not found in any stored image.")

    def clear_cache(self) -> None:
        """Clear the AI inference cache (in-memory and on-disk).

        Useful after the image library is significantly updated so that
        the next search always runs fresh inference rather than serving
        stale cached results.
        """
        self.ai_api.clear_cache()
        self.logger.info("Inference cache cleared via backend API")
