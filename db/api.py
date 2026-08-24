import logging
from pathlib import Path

from db.schema import ImageHistory, db

logging.basicConfig(
    level=logging.INFO, format="DB - %(asctime)s - %(levelname)s - %(message)s"
)


class db_engine:
    def __init__(self) -> None:
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.initialize()
        self.logger.info("DB Initialized and ready to go")

    def initialize(self) -> None:
        """Connect to the database and create tables if they do not already exist."""
        db.connect(reuse_if_open=True)
        db.create_tables([ImageHistory])
        self.logger.info("DB Connection Established, and Tables Ready")

    def finalize(self) -> None:
        """Close the database connection."""
        db.close()

    def add_image(self, imgpath: str) -> None:
        """Insert a new image record into the database.

        The file at imgpath must exist on disk before calling this method.
        """
        assert isinstance(imgpath, str), "imgpath must be a string"
        assert Path(imgpath).exists(), f"Image {imgpath} does not exist"

        record: ImageHistory = ImageHistory.create(imgpath=imgpath)
        if record:
            self.logger.info(f"Image {imgpath} added to db")
        else:
            self.logger.warning(f"Failed to add Image {imgpath} to db")

    def get_all_images_desc(self) -> list[ImageHistory]:
        """Return all image records ordered by timestamp descending (newest first).

        This powers the reverse-chronological search used by the episodic
        memory pipeline.
        """
        results: list[ImageHistory] = list(
            ImageHistory.select().order_by(ImageHistory.timestamp.desc())
        )
        self.logger.info(f"Retrieved {len(results)} images from db (newest first)")
        return results
