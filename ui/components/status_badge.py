"""Status badge: a text label and a colour for each item status."""

from nicegui import ui

from ui.models import ItemStatus

STATUS_LABELS: dict[ItemStatus, str] = {
    ItemStatus.WAITING: "Waiting",
    ItemStatus.SENDING: "Sending",
    ItemStatus.STORED: "Stored",
    ItemStatus.DUPLICATE: "Already stored",
    ItemStatus.REJECTED: "Rejected",
    ItemStatus.FAILED: "Failed",
}


def status_badge(status: ItemStatus) -> ui.label:
    """Build a badge. The text is always shown, so colour is not the only signal."""
    assert isinstance(status, ItemStatus), "status must be an ItemStatus"
    return ui.label(STATUS_LABELS[status]).classes(f"up-badge up-badge--{status.value}")
