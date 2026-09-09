"""One file in the upload queue."""

from collections.abc import Callable

from nicegui import ui

from ui.components.status_badge import status_badge
from ui.models import ItemStatus, UploadItem


def format_size(size: int) -> str:
    """Return a short size text, for example `1.4 MB`."""
    assert size >= 0, "size must not be negative"
    value: float = float(size)
    for unit in ("B", "KB", "MB"):
        if value < 1024 or unit == "MB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


class UploadRow:
    """A list row for one item. Call `update` when the item changes."""

    def __init__(self, item: UploadItem, on_retry: Callable[[int], None]) -> None:
        assert item is not None and callable(on_retry), "item and on_retry required"
        self._item: UploadItem = item
        self._on_retry: Callable[[int], None] = on_retry
        self._root: ui.element = ui.element("li").classes("up-row")
        self.update()

    def update(self) -> None:
        """Draw the row again from the current state of the item."""
        item: UploadItem = self._item
        self._root.clear()
        with self._root:
            with ui.element("div").classes("up-row__main"):
                ui.label(item.name).classes("up-row__name").tooltip(item.name)
                detail: str = format_size(item.size)
                if item.message:
                    detail = f"{detail} - {item.message}"
                ui.label(detail).classes("up-row__meta")
            with ui.element("div").classes("up-row__side"):
                status_badge(item.status)
                if item.status == ItemStatus.FAILED and item.data is not None:
                    ui.button(
                        "Try again",
                        color=None,
                        on_click=lambda: self._on_retry(item.id),
                    ).props("unelevated no-caps").classes("up-btn up-btn--secondary")
