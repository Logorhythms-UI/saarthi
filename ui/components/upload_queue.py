"""The list of files in the upload queue."""

from collections.abc import Callable

from nicegui import ui

from ui.components.upload_row import UploadRow
from ui.models import UploadItem

EMPTY_TEXT: str = "No files yet. Added files appear here."


class QueueView:
    """The list and its empty state. Rows update one by one."""

    def __init__(self, on_retry: Callable[[int], None]) -> None:
        assert callable(on_retry), "on_retry must be callable"
        self._on_retry: Callable[[int], None] = on_retry
        self._rows: dict[int, UploadRow] = {}
        with ui.element("div").classes("up-queue"):
            self._empty: ui.label = ui.label(EMPTY_TEXT).classes("up-queue__empty")
            self._list: ui.element = ui.element("ul").classes("up-queue__list")

    def render(self, items: list[UploadItem]) -> None:
        """Draw the whole list. Call it when an item is added or removed."""
        self._list.clear()
        self._rows = {}
        with self._list:
            for item in items:
                self._rows[item.id] = UploadRow(item, self._on_retry)
        self._empty.set_visibility(not items)

    def update_item(self, item: UploadItem) -> None:
        """Draw one row again."""
        row: UploadRow | None = self._rows.get(item.id)
        if row is not None:
            row.update()
