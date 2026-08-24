import logging
import uuid
import zipfile
from pathlib import Path
from shutil import copy

logger: logging.Logger = logging.getLogger(__name__)

_VALID_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
)


class StorageEngine:
    """Manages physical image file storage."""

    def __init__(self) -> None:
        # Deterministic path: always storage/uploads/ relative to this file
        self._upload_path: Path = Path(__file__).parent / "uploads"
        self._upload_path.mkdir(parents=True, exist_ok=True)
        logger.info("Storage engine ready — uploads at %s", self._upload_path)

    def store_upload(self, file_content: bytes, filename: str) -> str:
        """Store uploaded file content with a UUID-prefixed name.

        Prevents filename collisions from multiple uploads.
        Returns the absolute path of the stored file.
        """
        assert isinstance(file_content, bytes), "file_content must be bytes"
        assert isinstance(filename, str), "filename must be a string"
        assert len(file_content) > 0, "file_content must not be empty"

        extension: str = Path(filename).suffix.lower()
        if extension not in _VALID_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {extension}. "
                f"Supported: {', '.join(sorted(_VALID_IMAGE_EXTENSIONS))}"
            )

        safe_name: str = f"{uuid.uuid4().hex}_{Path(filename).stem}{extension}"
        dest: Path = self._upload_path / safe_name
        dest.write_bytes(file_content)

        logger.info("Stored upload: %s → %s", filename, safe_name)
        return str(dest)

    def upload_image(self, image_path: str) -> str:
        """Copy a single image into the uploads directory.

        Returns the absolute path of the stored image.
        """
        assert isinstance(image_path, str), "image_path must be a string"
        src: Path = Path(image_path)
        assert src.exists(), f"Image {image_path} does not exist"

        dest: Path = self._upload_path / src.name
        copy(str(src), str(dest))
        logger.info("Image %s uploaded successfully", src.name)
        return str(dest)

    def extract_zip(self, zip_path: str) -> list[str]:
        """Extract image files from a zip archive into the uploads directory.

        Only files with valid image extensions are extracted.  Returns a list
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
                    logger.info("Skipping non-image file: %s", member)
                    continue

                # Extract to uploads dir, using only the basename to flatten
                dest: Path = self._upload_path / member_path.name
                with zf.open(member) as source, open(str(dest), "wb") as target:
                    target.write(source.read())

                extracted_paths.append(str(dest))
                logger.info("Extracted: %s", member_path.name)

        extracted_paths.sort()
        logger.info("Extracted %d images from %s", len(extracted_paths), src.name)
        return extracted_paths

    def get_image_path(self, filename: str) -> str:
        """Resolve a filename to its absolute path inside the uploads directory."""
        assert isinstance(filename, str), "filename must be a string"
        full_path: Path = self._upload_path / filename
        assert full_path.exists(), f"Image {filename} not found in uploads"
        return str(full_path)
