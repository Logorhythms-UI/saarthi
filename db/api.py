import datetime
import logging
from pathlib import Path

from peewee import fn

from db.schema import ImageHistory, ObjectObservation, db

logger: logging.Logger = logging.getLogger(__name__)


class DatabaseEngine:
    """Manages image and object-observation metadata persistence using Peewee/SQLite."""

    def __init__(self) -> None:
        self._initialize()
        logger.info("Database initialized and ready")

    def _initialize(self) -> None:
        """Connect to the database, ensure schema is up to date, then close.

        Closing after init ensures tables are flushed to disk.
        Peewee's autoconnect (enabled by default) will open fresh
        connections from any thread that subsequently accesses the DB.
        """
        db.connect(reuse_if_open=True)
        db.create_tables([ImageHistory, ObjectObservation], safe=True)

        # Migrate: add sequence_number column if missing (existing DB)
        columns: list[str] = [col.name for col in db.get_columns("imagehistory")]
        if "sequence_number" not in columns:
            db.execute_sql(
                "ALTER TABLE imagehistory ADD COLUMN sequence_number INTEGER DEFAULT 0"
            )
            logger.info("Migrated: added sequence_number column")

        db.close()
        logger.info("Database schema ready (connection closed, autoconnect enabled)")

    def finalize(self) -> None:
        """Close the database connection."""
        db.close()

    def add_image(self, image_path: str) -> ImageHistory:
        """Insert a new image record with the next sequence number.

        The file at image_path must exist on disk.
        Returns the created ImageHistory record.
        """
        assert isinstance(image_path, str), "image_path must be a string"
        assert Path(image_path).exists(), f"Image {image_path} does not exist"

        next_seq: int = self._next_sequence_number()
        record: ImageHistory = ImageHistory.create(
            imgpath=image_path, sequence_number=next_seq
        )
        logger.info(
            "Image added to database: %s (sequence: %d)",
            Path(image_path).name,
            next_seq,
        )
        return record

    def add_object_observation(
        self,
        image_id: int,
        label: str,
        detector_confidence: float,
        bounding_box: str,
        timestamp: datetime.datetime,
    ) -> int:
        """Insert an ObjectObservation row and return its integer primary key.

        The primary key is used as the FAISS row ID in the semantic and visual indexes.

        Args:
            image_id: Primary key of the parent ImageHistory record.
            label: Broad-detector class label for the object.
            detector_confidence: Broad-detector confidence score in [0, 1].
            bounding_box: JSON-serialized [x1, y1, x2, y2] pixel coordinates.
            timestamp: Copied from the parent ImageHistory record.

        Returns:
            Integer primary key of the newly created ObjectObservation.
        """
        assert (
            isinstance(image_id, int) and image_id > 0
        ), "image_id must be a positive int"
        assert isinstance(label, str) and label, "label must be a non-empty string"
        assert (
            0.0 <= detector_confidence <= 1.0
        ), "detector_confidence must be in [0, 1]"
        assert (
            isinstance(bounding_box, str) and bounding_box
        ), "bounding_box must be a non-empty JSON string"
        assert isinstance(timestamp, datetime.datetime), "timestamp must be a datetime"

        obs: ObjectObservation = ObjectObservation.create(
            image_id=image_id,
            label=label,
            detector_confidence=detector_confidence,
            bounding_box=bounding_box,
            timestamp=timestamp,
        )
        return int(obs.id)

    def get_object_observations_by_ids(self, ids: list[int]) -> list[ObjectObservation]:
        """Batch-fetch ObjectObservation records by primary key list.

        Returns records in an unspecified order. Returns an empty list
        if ids is empty.
        """
        assert isinstance(ids, list), "ids must be a list"
        if not ids:
            return []
        return list(ObjectObservation.select().where(ObjectObservation.id.in_(ids)))

    def get_image_by_id(self, image_id: int) -> ImageHistory | None:
        """Fetch an ImageHistory record by primary key.

        Returns None if no record exists with that id.
        """
        assert (
            isinstance(image_id, int) and image_id > 0
        ), "image_id must be a positive int"
        return ImageHistory.get_or_none(ImageHistory.id == image_id)

    def get_all_images_desc(self) -> list[ImageHistory]:
        """Return all image records ordered newest-first by sequence number.

        Powers the reverse-chronological search pipeline.
        """
        results: list[ImageHistory] = list(
            ImageHistory.select().order_by(ImageHistory.sequence_number.desc())
        )
        logger.info("Retrieved %d images from database (newest first)", len(results))
        return results

    def get_image_count(self) -> int:
        """Return the total number of stored image records."""
        count: int = ImageHistory.select().count()
        return count

    def _next_sequence_number(self) -> int:
        """Compute the next monotonically increasing sequence number."""
        max_seq = ImageHistory.select(fn.MAX(ImageHistory.sequence_number)).scalar()
        if max_seq is None:
            return 1
        return max_seq + 1

    def delete_image(self, image_id: int) -> list[int]:
        """Delete an ImageHistory record and its associated ObjectObservations.

        Args:
            image_id: Primary key of the ImageHistory record.

        Returns:
            List of ObjectObservation primary keys that were deleted.
        """
        assert (
            isinstance(image_id, int) and image_id > 0
        ), "image_id must be a positive int"

        # 1. Get the object IDs associated with this image before deletion
        # so we can pass them to FAISS for removal.
        obs_ids = [
            obs.id
            for obs in ObjectObservation.select(ObjectObservation.id).where(
                ObjectObservation.image_id == image_id
            )
        ]

        # 2. Delete the image (ObjectObservations will be cascade deleted)
        record = ImageHistory.get_or_none(ImageHistory.id == image_id)
        if record:
            record.delete_instance()
            logger.info(
                "Deleted image (id=%d) and %d observations", image_id, len(obs_ids)
            )

        return obs_ids

    def clear_all(self) -> None:
        """Clear all ImageHistory and ObjectObservation records from the database."""
        ObjectObservation.delete().execute()
        ImageHistory.delete().execute()
        logger.info("Database cleared of all image and observation records")
