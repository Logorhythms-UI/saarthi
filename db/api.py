import logging
from pathlib import Path

from schema import ImageHistory, db
# from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="DB - %(asctime)s - %(levelname)s - %(message)s"
)


class db_engine:
    def __init__(self) -> None:
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.initialize()
        self.logger.info("DB Initialized and ready to go")
    
    def initialize(self) -> None:
        # Initialize the DB and create necessary tables
        db.connect()
        db.create_tables(
            [ImageHistory]
        )  # Safe creates the table (does not re-create if table already there)
        self.logger.info("Db Connection Esthablished, and Tables Ready")

    def finalize(self) -> None:
        db.close()

    def add_image(self, imgpath: str) -> None:
        assert Path(imgpath).exists(), f"Image {imgpath} does not exist"
        record = ImageHistory.create(imgpath=imgpath)
        if record:
            self.logger.info(f"Image {imgpath} added to db")
        else:
            self.logger.warning(f"Failed to add Image {imgpath} to db")
