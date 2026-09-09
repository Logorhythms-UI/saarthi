"""Page registration. The application calls this once at start-up."""

import logging

from nicegui import ui

from ui.pages.upload_page import build_upload_page
from ui.ports import UploadService
from ui.theme import register_static

logger: logging.Logger = logging.getLogger(__name__)

UPLOAD_ROUTE: str = "/upload"


def register_pages(service: UploadService) -> None:
    """Register the pages of the UI. The caller starts the server."""
    assert service is not None, "service is required"
    register_static()

    @ui.page(UPLOAD_ROUTE)
    def upload() -> None:
        build_upload_page(service)

    logger.info("Registered page %s", UPLOAD_ROUTE)
