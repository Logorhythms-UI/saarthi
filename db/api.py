import logging
from pathlib import Path

from peewee import fn

from db.schema import ImageHistory, db

logger: logging.Logger = logging.getLogger(__name__)


class DatabaseEngine:
    """Manages image metadata persistence using Peewee/SQLite."""

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
        db.create_tables([ImageHistory])

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

    def add_image(self, image_path: str) -> None:
        """Insert a new image record with the next sequence number.

        The file at image_path must exist on disk.
        """
        assert isinstance(image_path, str), "image_path must be a string"
        assert Path(image_path).exists(), f"Image {image_path} does not exist"

        next_seq: int = self._next_sequence_number()
        ImageHistory.create(imgpath=image_path, sequence_number=next_seq)
        logger.info(
            "Image added to database: %s (sequence: %d)",
            Path(image_path).name,
            next_seq,
        )

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
