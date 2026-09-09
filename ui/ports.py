"""Interface between the UI and the rest of the application.

The UI imports only this module to reach the backend. The backend phase
gives a real implementation of `UploadService`.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class OutcomeStatus(StrEnum):
    """Result of one upload."""

    STORED = "stored"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True)
class UploadOutcome:
    """Result of one upload, with a short text for the user."""

    status: OutcomeStatus
    # Must not contain personal data or internal error text.
    message: str


class UploadService(Protocol):
    """Accepts one image at a time."""

    async def submit_image(
        self, filename: str, content_type: str, data: bytes
    ) -> UploadOutcome:
        """Store one image. Return the outcome. Do not raise for expected cases."""
        ...
