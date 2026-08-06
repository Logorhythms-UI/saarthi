import logging
from pathlib import Path

from storage.app import storage_engine
from db.api import db_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="BACKEND - %(asctime)s - %(levelname)s - %(message)s"
)

class backend_api:
    def __init__(self):
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.storage_api = storage_engine()
        self.db_api = db_engine()
        self.logger.info("Backend API Started and ready to go")

    def upload_image(self, imgpath: str) -> None:
        self.storage_api.upload_image(imgpath)
        self.db_api.add_image(imgpath)
        self.logger.info(f"Uploaded Image to System Successfully")