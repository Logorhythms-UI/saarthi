"""Fake upload service for work on the UI before the backend exists.

The file name selects the outcome:
`duplicate` -> DUPLICATE, `reject` -> REJECTED, `fail` -> FAILED.
Any other name -> STORED.
"""

import asyncio
import logging
import random

from ui.ports import OutcomeStatus, UploadOutcome

logger: logging.Logger = logging.getLogger(__name__)

_OUTCOMES_BY_KEYWORD: dict[str, UploadOutcome] = {
    "duplicate": UploadOutcome(
        OutcomeStatus.DUPLICATE, "This image is already stored."
    ),
    "reject": UploadOutcome(OutcomeStatus.REJECTED, "The image is too blurry."),
    "fail": UploadOutcome(
        OutcomeStatus.FAILED, "The server could not store the image."
    ),
}
_STORED: UploadOutcome = UploadOutcome(OutcomeStatus.STORED, "")


class StubUploadService:
    """Implements `UploadService` with a short delay and a fixed outcome."""

    async def submit_image(
        self, filename: str, content_type: str, data: bytes
    ) -> UploadOutcome:
        assert isinstance(filename, str) and filename, "filename must be a string"
        assert isinstance(content_type, str), "content_type must be a string"
        assert isinstance(data, bytes) and data, "data must be non-empty bytes"

        await asyncio.sleep(random.uniform(0.4, 1.2))
        lowered: str = filename.lower()
        for keyword, outcome in _OUTCOMES_BY_KEYWORD.items():
            if keyword in lowered:
                logger.info("Stub outcome %s for keyword %s", outcome.status, keyword)
                return outcome
        return _STORED
