"""Summary of the queue and the button that clears finished files."""

from collections.abc import Callable, Mapping

from nicegui import ui

from ui.models import ItemStatus

CLEAR_LABEL: str = "Clear finished"


def summary_text(counts: Mapping[ItemStatus, int]) -> str:
    """Return a short text such as `3 of 5 stored, 1 failed`."""
    total: int = sum(counts.values())
    text: str = f"{counts.get(ItemStatus.STORED, 0)} of {total} stored"
    for status in (ItemStatus.DUPLICATE, ItemStatus.REJECTED, ItemStatus.FAILED):
        if counts.get(status, 0):
            text += f", {counts[status]} {status.value}"
    return text


class SummaryBar:
    """Shows the counts. Hidden while the queue is empty."""

    def __init__(self, on_clear: Callable[[], None]) -> None:
        assert callable(on_clear), "on_clear must be callable"
        with ui.element("div").classes("up-summary") as self._root:
            # A status region: screen readers announce changes.
            self._text: ui.label = (
                ui.label()
                .classes("up-summary__text")
                .props("role=status aria-live=polite")
            )
            self._clear: ui.button = (
                ui.button(CLEAR_LABEL, color=None, on_click=on_clear)
                .props("unelevated no-caps")
                .classes("up-btn up-btn--secondary")
            )
        self.update({}, finished=0)

    def update(self, counts: Mapping[ItemStatus, int], finished: int) -> None:
        """Show new counts. `finished` is the number of items that can be cleared."""
        assert finished >= 0, "finished must not be negative"
        total: int = sum(counts.values())
        self._root.set_visibility(total > 0)
        self._text.set_text(summary_text(counts))
        self._clear.set_visibility(finished > 0)
