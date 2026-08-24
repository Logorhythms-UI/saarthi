import logging
import zipfile
from pathlib import Path
from shutil import copy

logging.basicConfig(
    level=logging.INFO, format="STORAGE - %(asctime)s - %(levelname)s - %(message)s"
)

# Image extensions considered valid during zip extraction
_VALID_IMAGE_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class storage_engine:
    def __init__(self) -> None:
        self.logger: logging.Logger = logging.getLogger(__name__)
        # Deterministic path: always storage/uploads/ relative to this file
        self.upload_path: Path = Path(__file__).parent / "uploads"
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Storage Engine Ready — uploads at {self.upload_path}")

    def upload_image(self, imgurl: str) -> str:
        """Copy a single image into the uploads directory.

        Returns the absolute path of the stored image.
        """
        assert isinstance(imgurl, str), "imgurl must be a string"
        src: Path = Path(imgurl)
        assert src.exists(), f"Image {imgurl} does not exist"

        dest: Path = self.upload_path / src.name
        copy(str(src), str(dest))
        self.logger.info(f"Image {src.name} uploaded successfully")
        return str(dest)

    def extract_zip(self, zip_path: str) -> list[str]:
        """Extract image files from a zip archive into the uploads directory.

        Only files with valid image extensions are extracted. Returns a list
        of absolute paths for the extracted images, sorted alphabetically.
        """
        assert isinstance(zip_path, str), "zip_path must be a string"
        src: Path = Path(zip_path)
        assert src.exists(), f"Zip file {zip_path} does not exist"
        assert zipfile.is_zipfile(str(src)), f"{zip_path} is not a valid zip file"

        extracted_paths: list[str] = []

        with zipfile.ZipFile(str(src), "r") as zf:
            for member in zf.namelist():
                member_path: Path = Path(member)
                # Skip directories and hidden/macOS resource-fork files
                if member.endswith("/") or member_path.name.startswith("."):
                    continue
                if member_path.suffix.lower() not in _VALID_IMAGE_EXTENSIONS:
                    self.logger.info(f"Skipping non-image file: {member}")
                    continue

                # Extract to uploads dir, using only the basename to flatten
                dest: Path = self.upload_path / member_path.name
                with zf.open(member) as source, open(str(dest), "wb") as target:
                    target.write(source.read())

                extracted_paths.append(str(dest))
                self.logger.info(f"Extracted: {member_path.name}")

        extracted_paths.sort()
        self.logger.info(f"Extracted {len(extracted_paths)} images from {src.name}")
        return extracted_paths

    def get_image_path(self, filename: str) -> str:
        """Resolve a filename to its absolute path inside the uploads directory.

        Asserts that the resolved file actually exists.
        """
        assert isinstance(filename, str), "filename must be a string"
        full_path: Path = self.upload_path / filename
        assert full_path.exists(), f"Image {filename} not found in uploads"
        return str(full_path)
