import dataclasses
import hashlib
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from storage.errors import ImageTooLargeError, UnsupportedImageError
from storage.imaging import PreparedImage, prepare_image
from storage.settings import StorageConfig, load_config

CONFIG: StorageConfig = load_config()

# JPEG markers that a clean file may contain: SOI, APP0 (JFIF), DQT, SOF, DHT, SOS, DRI, EOI.
ALLOWED_MARKERS: set[int] = {0xD8, 0xE0, 0xDB, 0xC0, 0xC1, 0xC2, 0xC4, 0xDA, 0xDD, 0xD9}
SECRET_TEXT: bytes = b"secret-note-12345"


def picture(size: tuple[int, int] = (200, 100), seed: int = 0) -> Image.Image:
    rng: np.random.Generator = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def encode(image: Image.Image, fmt: str = "JPEG", **kwargs: object) -> bytes:
    buffer: BytesIO = BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def jpeg_markers(data: bytes) -> list[int]:
    """Return the marker codes of a JPEG file, up to the start of the scan."""
    assert data[:2] == b"\xff\xd8"
    markers: list[int] = [0xD8]
    position: int = 2
    while position < len(data):
        assert data[position] == 0xFF, "marker expected"
        code: int = data[position + 1]
        markers.append(code)
        if code == 0xDA:  # Start of scan: the image data follows
            break
        length: int = int.from_bytes(data[position + 2 : position + 4], "big")
        position += 2 + length
    return markers


def dirty_jpeg() -> bytes:
    """A JPEG with orientation, GPS, a comment, an ICC profile and XMP."""
    exif: Image.Exif = Image.Exif()
    exif[0x0112] = 6  # Orientation
    exif[0x010F] = "Secret Camera Maker"
    exif[0x8825] = {1: "N", 2: (12.0, 58.0, 30.0), 3: "E", 4: (77.0, 35.0, 10.0)}  # GPS
    return encode(
        picture(),
        exif=exif.tobytes(),
        comment=SECRET_TEXT,
        icc_profile=b"ICCPROFILE-SECRET" * 20,
        xmp=b"<x:xmpmeta>" + SECRET_TEXT + b"</x:xmpmeta>",
    )


def test_input_has_the_metadata_that_the_tests_expect() -> None:
    data: bytes = dirty_jpeg()
    assert b"Exif" in data and SECRET_TEXT in data and b"ICC_PROFILE" in data
    exif: Image.Exif = Image.open(BytesIO(data)).getexif()
    assert exif[0x0112] == 6
    assert exif.get_ifd(0x8825)[1] == "N"  # The GPS data is there


def test_all_metadata_is_removed_and_the_orientation_is_applied() -> None:
    prepared: PreparedImage = prepare_image(dirty_jpeg(), CONFIG)
    assert (prepared.width, prepared.height) == (100, 200)  # Turned by the orientation
    assert set(jpeg_markers(prepared.data)) <= ALLOWED_MARKERS
    for text in (
        b"Exif",
        SECRET_TEXT,
        b"ICC_PROFILE",
        b"xmpmeta",
        b"Photoshop",
        b"Secret Camera",
    ):
        assert text not in prepared.data
    saved: Image.Image = Image.open(BytesIO(prepared.data))
    assert len(saved.getexif()) == 0 and not saved.getexif().get_ifd(0x8825)
    assert saved.format == "JPEG" and saved.mode == "RGB"
    assert "icc_profile" not in saved.info and "comment" not in saved.info


def test_reported_size_and_hash_match_the_output() -> None:
    prepared: PreparedImage = prepare_image(encode(picture((320, 240))), CONFIG)
    assert (
        Image.open(BytesIO(prepared.data)).size
        == (prepared.width, prepared.height)
        == (320, 240)
    )
    assert prepared.sha256 == hashlib.sha256(prepared.data).hexdigest()


def test_same_input_gives_the_same_hash_and_other_input_a_different_one() -> None:
    data: bytes = encode(picture())
    assert prepare_image(data, CONFIG).sha256 == prepare_image(data, CONFIG).sha256
    assert (
        prepare_image(encode(picture(seed=1)), CONFIG).sha256
        != prepare_image(data, CONFIG).sha256
    )


def test_metadata_does_not_change_the_hash_of_the_same_picture() -> None:
    plain: bytes = encode(picture((100, 200)))
    exif: Image.Exif = Image.Exif()
    exif[0x010F] = "Another Maker"
    with_exif: bytes = encode(picture((100, 200)), exif=exif.tobytes())
    assert (
        prepare_image(plain, CONFIG).sha256 == prepare_image(with_exif, CONFIG).sha256
    )


def test_png_with_alpha_gets_a_white_background() -> None:
    rgba: Image.Image = Image.new("RGBA", (40, 40), (0, 0, 0, 0))  # Fully transparent
    prepared: PreparedImage = prepare_image(encode(rgba, "PNG"), CONFIG)
    pixel: tuple[int, ...] = Image.open(BytesIO(prepared.data)).getpixel((20, 20))
    assert min(pixel) >= 250


@pytest.mark.parametrize(
    "image",
    [
        picture((50, 40)).convert("L"),
        picture((50, 40)).convert("CMYK"),
        picture((50, 40)).convert("P"),
    ],
)
def test_other_colour_modes_become_rgb_jpeg(image: Image.Image) -> None:
    fmt: str = "JPEG" if image.mode == "CMYK" else "PNG"
    prepared: PreparedImage = prepare_image(encode(image, fmt), CONFIG)
    assert Image.open(BytesIO(prepared.data)).mode == "RGB"


def test_webp_becomes_jpeg() -> None:
    prepared: PreparedImage = prepare_image(encode(picture(), "WEBP"), CONFIG)
    assert Image.open(BytesIO(prepared.data)).format == "JPEG"


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"not an image",
        encode(picture(), "GIF"),
        encode(picture(), "BMP"),
        encode(picture(), "TIFF"),
        b"\xff\xd8\xff\xe0" + b"x" * 40,
        encode(Image.new("I;16", (20, 20)), "PNG"),  # 16-bit
    ],
    ids=["empty", "text", "gif", "bmp", "tiff", "fake jpeg", "16 bit"],
)
def test_unusable_files_are_refused(data: bytes) -> None:
    with pytest.raises(UnsupportedImageError):
        prepare_image(data, CONFIG)


def test_truncated_file_is_refused() -> None:
    data: bytes = encode(picture((400, 300)))
    with pytest.raises(UnsupportedImageError):
        prepare_image(data[: len(data) // 2], CONFIG)


@pytest.mark.parametrize("fmt", ["WEBP", "PNG"])
def test_animated_images_are_refused(fmt: str) -> None:
    frames: list[Image.Image] = [picture((30, 30), seed=i) for i in range(3)]
    data: bytes = encode(
        frames[0], fmt, save_all=True, append_images=frames[1:], duration=100
    )
    with pytest.raises(UnsupportedImageError):
        prepare_image(data, CONFIG)


def test_too_many_bytes_and_too_many_pixels_are_refused() -> None:
    data: bytes = encode(picture((100, 100)))
    with pytest.raises(ImageTooLargeError):
        prepare_image(data, dataclasses.replace(CONFIG, max_bytes=len(data) - 1))
    with pytest.raises(ImageTooLargeError):
        prepare_image(data, dataclasses.replace(CONFIG, max_pixels=9_999))
    assert (
        prepare_image(data, dataclasses.replace(CONFIG, max_pixels=10_000)).width == 100
    )


def test_parameters_are_checked() -> None:
    with pytest.raises(AssertionError):
        prepare_image("text", CONFIG)  # type: ignore[arg-type]
