import asyncio
from collections.abc import Awaitable, Callable

from ui.models import ItemStatus, UploadItem
from ui.ports import OutcomeStatus, UploadOutcome
from ui.settings import MAX_FILE_SIZE_BYTES
from ui.upload_session import (
    MESSAGE_EMPTY,
    MESSAGE_SERVICE_ERROR,
    MESSAGE_TOO_LARGE,
    MESSAGE_UNSUPPORTED_TYPE,
    UploadSession,
)

JPEG: str = "image/jpeg"


class FakeService:
    """Records calls. Name prefixes select the outcome."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.active: int = 0
        self.max_active: int = 0

    async def submit_image(
        self, filename: str, content_type: str, data: bytes
    ) -> UploadOutcome:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append(filename)
        await asyncio.sleep(0)
        self.active -= 1
        if filename.startswith("boom"):
            raise RuntimeError("internal detail that must not reach the user")
        if filename.startswith("dup"):
            return UploadOutcome(OutcomeStatus.DUPLICATE, "already stored")
        if filename.startswith("bad"):
            return UploadOutcome(OutcomeStatus.REJECTED, "too blurry")
        if filename.startswith("fail"):
            return UploadOutcome(OutcomeStatus.FAILED, "server error")
        return UploadOutcome(OutcomeStatus.STORED, "")


def run(scenario: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(scenario())


def test_items_are_sent_in_order_one_at_a_time() -> None:
    async def scenario() -> None:
        service: FakeService = FakeService()
        session: UploadSession = UploadSession(service)
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            session.add(name, JPEG, b"data")
        await session.wait_idle()
        assert service.calls == ["a.jpg", "b.jpg", "c.jpg"]
        assert service.max_active == 1
        assert session.count(ItemStatus.STORED) == 3

    run(scenario)


def test_each_outcome_sets_the_status() -> None:
    async def scenario() -> None:
        session: UploadSession = UploadSession(FakeService())
        names: list[str] = ["ok.jpg", "dup.jpg", "bad.jpg", "fail.jpg"]
        for name in names:
            session.add(name, JPEG, b"data")
        await session.wait_idle()
        statuses: list[ItemStatus] = [i.status for i in session.items]
        assert statuses == [
            ItemStatus.STORED,
            ItemStatus.DUPLICATE,
            ItemStatus.REJECTED,
            ItemStatus.FAILED,
        ]
        assert session.items[1].message == "already stored"

    run(scenario)


def test_service_exception_fails_the_item_and_the_queue_goes_on() -> None:
    async def scenario() -> None:
        session: UploadSession = UploadSession(FakeService())
        session.add("boom.jpg", JPEG, b"data")
        session.add("next.jpg", JPEG, b"data")
        await session.wait_idle()
        first, second = session.items
        assert first.status == ItemStatus.FAILED
        assert first.message == MESSAGE_SERVICE_ERROR  # no internal text
        assert second.status == ItemStatus.STORED

    run(scenario)


def test_files_that_break_the_limits_are_rejected_without_a_service_call() -> None:
    async def scenario() -> None:
        service: FakeService = FakeService()
        session: UploadSession = UploadSession(service)
        session.add("empty.jpg", JPEG, b"")
        session.add("doc.pdf", "application/pdf", b"data")
        session.add("big.jpg", JPEG, b"x" * (MAX_FILE_SIZE_BYTES + 1))
        await session.wait_idle()
        assert service.calls == []
        assert [i.message for i in session.items] == [
            MESSAGE_EMPTY,
            MESSAGE_UNSUPPORTED_TYPE,
            MESSAGE_TOO_LARGE,
        ]
        assert all(i.status == ItemStatus.REJECTED for i in session.items)
        assert all(i.data is None for i in session.items)

    run(scenario)


def test_bytes_are_kept_only_for_failed_items() -> None:
    async def scenario() -> None:
        session: UploadSession = UploadSession(FakeService())
        session.add("ok.jpg", JPEG, b"data")
        session.add("fail.jpg", JPEG, b"data")
        await session.wait_idle()
        stored, failed = session.items
        assert stored.data is None
        assert failed.data == b"data"

    run(scenario)


def test_retry_sends_a_failed_item_again() -> None:
    async def scenario() -> None:
        service: FakeService = FakeService()
        session: UploadSession = UploadSession(service)
        item: UploadItem = session.add("fail.jpg", JPEG, b"data")
        await session.wait_idle()
        assert item.status == ItemStatus.FAILED
        session.retry(item.id)
        await session.wait_idle()
        assert service.calls == ["fail.jpg", "fail.jpg"]
        assert item.status == ItemStatus.FAILED  # the fake fails again

    run(scenario)


def test_retry_ignores_items_that_are_not_failed() -> None:
    async def scenario() -> None:
        service: FakeService = FakeService()
        session: UploadSession = UploadSession(service)
        item: UploadItem = session.add("ok.jpg", JPEG, b"data")
        await session.wait_idle()
        session.retry(item.id)
        session.retry(999)
        await session.wait_idle()
        assert service.calls == ["ok.jpg"]

    run(scenario)


def test_clear_finished_keeps_failed_items() -> None:
    async def scenario() -> None:
        session: UploadSession = UploadSession(FakeService())
        for name in ("ok.jpg", "dup.jpg", "bad.jpg", "fail.jpg"):
            session.add(name, JPEG, b"data")
        await session.wait_idle()
        assert session.clear_finished() == 3
        assert [i.name for i in session.items] == ["fail.jpg"]

    run(scenario)


def test_listeners_are_called_and_their_errors_do_not_stop_the_queue() -> None:
    async def scenario() -> None:
        seen: list[tuple[int, ItemStatus]] = []
        list_changes: list[int] = []

        def on_item(item: UploadItem) -> None:
            seen.append((item.id, item.status))
            raise ValueError("view error")

        session: UploadSession = UploadSession(
            FakeService(),
            on_item_changed=on_item,
            on_list_changed=lambda: list_changes.append(1),
        )
        session.add("a.jpg", JPEG, b"data")
        session.add("b.jpg", JPEG, b"data")
        await session.wait_idle()
        assert seen == [
            (1, ItemStatus.SENDING),
            (1, ItemStatus.STORED),
            (2, ItemStatus.SENDING),
            (2, ItemStatus.STORED),
        ]
        assert len(list_changes) == 2

    run(scenario)
