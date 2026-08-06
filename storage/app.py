import logging
from pathlib import Path
from shutil import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="STORAGE - %(asctime)s - %(levelname)s - %(message)s"
)

class storage_engine:
    def __init__(self) -> None:
        self.logger: logging.Logger = logging.getLogger(__name__)
        # Set the path for uploads here
        self.upload_path = "./uploads/"
        self.logger.info("Storage Engine Ready")

    def upload_image(self, imgurl: str) -> None:
        # Check Image exists
        assert Path(imgurl).exists(), f"Image ${imgurl} does not exist"

        copy(imgurl, self.upload_path)
        self.logger.info(f"Image {imgurl} uploaded successfully")
        
