"""Upload queue of one browser tab.

The session keeps the list of files and sends them to the service one at a
time. It has no NiceGUI code. The view reads the items and listens for changes.
"""

import asyncio
import logging
from collections.abc import Callable

from ui.models import FINISHED_STATUSES, ItemStatus, UploadItem
from ui.ports import UploadOutcome, UploadService
from ui.settings import ACCEPTED_CONTENT_TYPES, MAX_FILE_SIZE_BYTES

logger: logging.Logger = logging.getLogger(__name__)

MESSAGE_UNSUPPORTED_TYPE: str = "This file type is not supported."
MESSAGE_TOO_LARGE: str = "This file is too large."
MESSAGE_EMPTY: str = "This file is empty."
MESSAGE_SERVICE_ERROR: str = "The upload failed. Try again."


class UploadSession:
    """Sends the files of one tab to the service, one at a time.

    `on_item_changed` runs when one item changes. `on_list_changed` runs when
    an item is added or removed. A listener error is logged and does not stop
    the queue.
    """

    def __init__(
        self,
        service: UploadService,
        on_item_changed: Callable[[UploadItem], None] | None = None,
        on_list_changed: Callable[[], None] | None = None,
    ) -> None:
        assert service is not None, "service is required"
        self._service: UploadService = service
        self._on_item_changed = on_item_changed
        self._on_list_changed = on_list_changed
        self._items: list[UploadItem] = []
        self._next_id: int = 1
        self._worker: asyncio.Task[None] | None = None

    @property
    def items(self) -> list[UploadItem]:
        """Return a copy of the item list, oldest first."""
        return list(self._items)

    def count(self, status: ItemStatus) -> int:
        """Return how many items have this status."""
        assert isinstance(status, ItemStatus), "status must be an ItemStatus"
        return sum(1 for item in self._items if item.status == status)

    def add(self, name: str, content_type: str, data: bytes) -> UploadItem:
        """Add a file to the queue. Call it from a running event loop."""
        assert isinstance(name, str) and name, "name must be a non-empty string"
        assert isinstance(content_type, str), "content_type must be a string"
        assert isinstance(data, bytes), "data must be bytes"

        item: UploadItem = UploadItem(
            id=self._next_id, name=name, content_type=content_type, size=len(data)
        )
        self._next_id += 1
        problem: str | None = self._check_limits(item)
        if problem is None:
            item.data = data
        else:
            item.status = ItemStatus.REJECTED
            item.message = problem
            logger.info("Item %d rejected before sending: %s", item.id, problem)
        self._items.append(item)
        self._notify_list()
        self._ensure_worker()
        return item

    def retry(self, item_id: int) -> None:
        """Put a failed item back in the queue."""
        assert isinstance(item_id, int), "item_id must be an int"
        item: UploadItem | None = self._find(item_id)
        if item is None or item.status != ItemStatus.FAILED or item.data is None:
            logger.warning("Item %d cannot be retried", item_id)
            return
        item.status = ItemStatus.WAITING
        item.message = ""
        self._notify_item(item)
        self._ensure_worker()

    def clear_finished(self) -> int:
        """Remove the items that need no more action. Return how many."""
        before: int = len(self._items)
        self._items = [i for i in self._items if i.status not in FINISHED_STATUSES]
        removed: int = before - len(self._items)
        if removed:
            self._notify_list()
        return removed

    async def wait_idle(self) -> None:
        """Wait until the queue has no waiting or sending item."""
        while self._worker is not None and not self._worker.done():
            await asyncio.shield(self._worker)

    def _check_limits(self, item: UploadItem) -> str | None:
        if item.size == 0:
            return MESSAGE_EMPTY
        if item.content_type not in ACCEPTED_CONTENT_TYPES:
            return MESSAGE_UNSUPPORTED_TYPE
        if item.size > MAX_FILE_SIZE_BYTES:
            return MESSAGE_TOO_LARGE
        return None

    def _find(self, item_id: int) -> UploadItem | None:
        return next((i for i in self._items if i.id == item_id), None)

    def _next_waiting(self) -> UploadItem | None:
        return next((i for i in self._items if i.status == ItemStatus.WAITING), None)

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        item: UploadItem | None = self._next_waiting()
        while item is not None:
            await self._send(item)
            item = self._next_waiting()

    async def _send(self, item: UploadItem) -> None:
        assert item.data is not None, "a waiting item must have data"
        item.status = ItemStatus.SENDING
        self._notify_item(item)
        try:
            outcome: UploadOutcome = await self._service.submit_image(
                item.name, item.content_type, item.data
            )
            item.status = ItemStatus(outcome.status.value)
            item.message = outcome.message
        except Exception:  # The service is outside this module
            # One bad file must not stop the queue.
            logger.exception("Service error while sending item %d", item.id)
            item.status = ItemStatus.FAILED
            item.message = MESSAGE_SERVICE_ERROR
        if item.status in FINISHED_STATUSES:
            item.data = None
        logger.info("Item %d is now %s", item.id, item.status.value)
        self._notify_item(item)

    def _notify_item(self, item: UploadItem) -> None:
        if self._on_item_changed is None:
            return
        try:
            self._on_item_changed(item)
        except Exception:  # A view error must not stop the queue
            logger.exception("Listener error for item %d", item.id)

    def _notify_list(self) -> None:
        if self._on_list_changed is None:
            return
        try:
            self._on_list_changed()
        except Exception:  # A view error must not stop the queue
            logger.exception("Listener error for list change")
