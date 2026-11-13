"""State objects of the upload page."""

from dataclasses import dataclass, field
from enum import StrEnum


class ItemStatus(StrEnum):
    """State of one file in the upload queue."""

    WAITING = "waiting"  # In the queue, not sent yet
    SENDING = "sending"  # The service handles the file now
    STORED = "stored"  # The service accepted the file
    DUPLICATE = "duplicate"  # The service already has this image
    REJECTED = "rejected"  # The file cannot be used (type, size, content)
    FAILED = "failed"  # An error occurred. The user can try again


# Statuses after which the file bytes are not needed again.
FINISHED_STATUSES: frozenset[ItemStatus] = frozenset(
    {ItemStatus.STORED, ItemStatus.DUPLICATE, ItemStatus.REJECTED}
)


@dataclass
class UploadItem:
    """One file in the queue."""

    id: int
    name: str
    content_type: str
    size: int
    status: ItemStatus = ItemStatus.WAITING
    message: str = ""
    # Kept while the file is waiting, sending or failed. Dropped after that.
    data: bytes | None = field(default=None, repr=False)
