"""NiceGUI frontend for the Saarthi episodic memory application.

Provides four routes:
    /       — Landing page with navigation
    /upload — Single-image temporal upload
    /search — Natural-language object search with bounding-box results
    /images — Reverse-chronological image gallery
"""

import logging
from pathlib import Path

from nicegui import run, ui
from nicegui.events import UploadEventArguments

from backend.api import Backend
from utils.result import Result

logger: logging.Logger = logging.getLogger(__name__)

# Shared CSS injected into every page
_SHARED_CSS: str = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
        color: #1f2937;
        min-height: 100vh;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 32px;
        margin-bottom: 24px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.12);
    }
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .primary-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .primary-btn:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    .nav-btn {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        padding: 20px 32px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        color: #667eea !important;
        transition: all 0.3s ease !important;
    }
    .nav-btn:hover {
        background: rgba(102, 126, 234, 0.1) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2) !important;
    }
    .result-image {
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        max-width: 100%;
    }
    .status-msg {
        padding: 12px 16px;
        border-radius: 8px;
        font-weight: 500;
    }
    .status-success {
        background: rgba(16, 185, 129, 0.1);
        color: #059669;
    }
    .status-error {
        background: rgba(239, 68, 68, 0.1);
        color: #dc2626;
    }
    .status-info {
        background: rgba(102, 126, 234, 0.1);
        color: #667eea;
    }
    .metadata-chip {
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 0.9rem;
        color: #4c51bf;
    }
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 20px;
    }
    .image-tile {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.5);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        cursor: default;
    }
    .image-tile:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 32px rgba(0, 0, 0, 0.13);
    }
    .image-tile img {
        width: 100%;
        height: 180px;
        object-fit: cover;
        display: block;
    }
    .image-tile-meta {
        padding: 10px 14px;
    }
    .image-tile-seq {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #667eea;
        margin-bottom: 2px;
    }
    .image-tile-name {
        font-size: 0.85rem;
        font-weight: 600;
        color: #1f2937;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 2px;
    }
    .image-tile-ts {
        font-size: 0.75rem;
        color: #9ca3af;
    }
    .empty-state {
        text-align: center;
        padding: 60px 24px;
        color: #9ca3af;
    }
    .empty-state-icon {
        font-size: 3.5rem;
        margin-bottom: 12px;
    }
</style>
"""


class Frontend:
    """NiceGUI web frontend for Saarthi."""

    def __init__(self, backend: Backend) -> None:
        assert isinstance(backend, Backend), "backend must be a Backend instance"
        self._backend: Backend = backend
        logger.info("Frontend initialized")

    def setup_routes(self) -> None:
        """Register all NiceGUI page routes."""
        self._register_landing_page()
        self._register_upload_page()
        self._register_search_page()
        self._register_images_page()
        logger.info("All routes registered: /, /upload, /search, /images")

    def _register_landing_page(self) -> None:
        """Register the / landing page route."""

        @ui.page("/")
        def _landing_page() -> None:
            _inject_css()
            with ui.column().classes(
                "w-full max-w-3xl mx-auto p-8 items-center justify-center"
            ):
                ui.label("Saarthi").classes(
                    "text-6xl font-extrabold mb-2 gradient-text tracking-tight"
                )
                ui.label("Episodic Memory Aid").classes(
                    "text-xl mb-12 text-gray-500 font-medium"
                )

                with ui.row().classes("gap-6 flex-wrap justify-center"):
                    ui.button(
                        "Upload Images",
                        on_click=lambda: ui.navigate.to("/upload"),
                    ).classes("nav-btn")
                    ui.button(
                        "Search Objects",
                        on_click=lambda: ui.navigate.to("/search"),
                    ).classes("nav-btn")
                    ui.button(
                        "Browse Memory",
                        on_click=lambda: ui.navigate.to("/images"),
                    ).classes("nav-btn")

    def _register_upload_page(self) -> None:
        """Register the /upload page route."""
        engine: Frontend = self

        @ui.page("/upload")
        def _upload_page() -> None:
            _inject_css()
            with ui.column().classes("w-full max-w-3xl mx-auto p-6 items-stretch"):
                # Header
                with ui.row().classes("items-center mb-6"):
                    ui.button(
                        icon="arrow_back",
                        on_click=lambda: ui.navigate.to("/"),
                    ).props("flat round")
                    ui.label("Upload Images").classes(
                        "text-3xl font-bold gradient-text"
                    )

                with ui.card().classes("glass-card w-full"):
                    ui.label(
                        "Upload images one at a time to simulate wearable capture. "
                        "Each image becomes the newest memory."
                    ).classes("text-gray-500 mb-2")
                    ui.label("Supported: JPG, JPEG, PNG, BMP, WebP, TIFF").classes(
                        "text-sm text-gray-400 mb-4"
                    )

                    image_count: int = engine._backend.get_image_count()
                    count_label: ui.label = ui.label(
                        f"{image_count} image(s) in memory."
                    ).classes("status-msg status-info mb-4")

                    upload_status: ui.label = ui.label("")
                    upload_status.set_visibility(False)

                    async def handle_upload(e: UploadEventArguments) -> None:
                        filename: str = e.file.name or "unknown"
                        logger.info("Received upload: %s", filename)

                        # SmallFileUpload.read() is a coroutine in nicegui
                        file_content: bytes = await e.file.read()

                        result: Result = await run.io_bound(
                            engine._backend.ingest_upload, file_content, filename
                        )

                        if result.ok:
                            current_count: int = engine._backend.get_image_count()
                            count_label.set_text(f"{current_count} image(s) in memory.")
                            upload_status.set_text(
                                f"✓ {filename} uploaded successfully"
                            )
                            upload_status.classes(
                                remove="status-error", add="status-success status-msg"
                            )
                            upload_status.set_visibility(True)
                            ui.notify(f"Ingested {filename}", type="positive")
                        else:
                            upload_status.set_text(f"✗ Upload failed: {result.err}")
                            upload_status.classes(
                                remove="status-success", add="status-error status-msg"
                            )
                            upload_status.set_visibility(True)
                            ui.notify(f"Upload failed: {result.err}", type="negative")

                    ui.upload(
                        on_upload=handle_upload,
                        auto_upload=True,
                    ).classes(
                        "w-full border-2 border-dashed border-gray-300 rounded-xl "
                        "p-4 bg-gray-50/50 hover:bg-gray-50 transition-colors"
                    ).props('accept=".jpg,.jpeg,.png,.bmp,.webp,.tiff"')

    def _register_search_page(self) -> None:
        """Register the /search page route."""
        engine: Frontend = self

        @ui.page("/search")
        def _search_page() -> None:
            _inject_css()
            with ui.column().classes("w-full max-w-4xl mx-auto p-6 items-stretch"):
                # Header
                with ui.row().classes("items-center mb-6"):
                    ui.button(
                        icon="arrow_back",
                        on_click=lambda: ui.navigate.to("/"),
                    ).props("flat round")
                    ui.label("Search for an Object").classes(
                        "text-3xl font-bold gradient-text"
                    )

                with ui.card().classes("glass-card w-full"):
                    ui.label(
                        "Describe what you're looking for in natural language. "
                        "The system will normalise your query and search the "
                        "indexed object memory."
                    ).classes("text-gray-500 mb-4")

                    query_input: ui.input = ui.input(
                        label="What are you looking for?",
                        placeholder=("e.g., I can't find my red backpack"),
                    ).classes("w-full text-lg mb-4")

                    results_container: ui.column = ui.column().classes(
                        "w-full items-stretch gap-4"
                    )

                    async def handle_search() -> None:
                        query: str = query_input.value.strip()
                        if not query:
                            ui.notify("Please enter a search query.", type="warning")
                            return

                        logger.info("Search requested: '%s'", query)
                        results_container.clear()

                        with results_container:
                            ui.spinner("dots", size="lg").classes("mx-auto my-4")
                            ui.label("Searching...").classes(
                                "text-center text-gray-500 font-medium"
                            )

                        result: Result = await run.io_bound(
                            engine._backend.search, query
                        )

                        results_container.clear()
                        with results_container:
                            if not result.ok:
                                ui.label(result.err).classes(
                                    "text-lg text-center status-msg status-error"
                                )
                            elif result.result["status"] == "FOUND":
                                _render_found_result(result.result)
                            else:
                                _render_not_found_result(result.result)

                    ui.button("Search", on_click=handle_search).classes(
                        "primary-btn w-full text-lg"
                    )

    def _register_images_page(self) -> None:
        """Register the /images gallery page route."""
        engine: Frontend = self

        @ui.page("/images")
        def _images_page() -> None:
            _inject_css()
            with ui.column().classes("w-full max-w-6xl mx-auto p-6 items-stretch"):
                with ui.row().classes("items-center justify-between mb-6 w-full"):
                    with ui.row().classes("items-center"):
                        ui.button(
                            icon="arrow_back",
                            on_click=lambda: ui.navigate.to("/"),
                        ).props("flat round")
                        ui.label("Memory Browser").classes(
                            "text-3xl font-bold gradient-text"
                        )

                    images: list = engine._backend.get_all_images_desc()

                    if images:

                        async def confirm_clear() -> None:
                            with ui.dialog() as dialog, ui.card():
                                ui.label("Are you sure?").classes(
                                    "text-xl font-bold mb-2"
                                )
                                ui.label(
                                    "This will permanently delete all images and memory."
                                ).classes("mb-4")
                                with ui.row():
                                    ui.button("Cancel", on_click=dialog.close).props(
                                        "flat"
                                    )

                                    def perform_clear() -> None:
                                        result = engine._backend.clear_all_memory()
                                        if result.ok:
                                            ui.notify(
                                                "All memory cleared", type="positive"
                                            )
                                            dialog.close()
                                            ui.navigate.to("/images")
                                        else:
                                            ui.notify(result.err, type="negative")

                                    ui.button(
                                        "Clear All",
                                        color="negative",
                                        on_click=perform_clear,
                                    )
                            dialog.open()

                        ui.button(
                            "Clear All", on_click=confirm_clear, color="red-5"
                        ).classes("font-semibold rounded-lg")

                if not images:
                    with ui.element("div").classes("empty-state w-full"):
                        ui.element("div").classes("empty-state-icon").style(
                            "font-size: 3.5rem"
                        )
                        ui.label("📭").style("font-size: 3.5rem; display: block")
                        ui.label("No images in memory yet.").classes(
                            "text-xl font-semibold mt-2"
                        )
                        ui.label(
                            "Upload some images first using the Upload page."
                        ).classes("text-sm mt-1")
                    return

                total: int = len(images)
                ui.label(f"{total} image(s) stored — newest first").classes(
                    "text-sm text-gray-400 mb-4"
                )

                with ui.element("div").classes("image-grid w-full"):
                    for rank, record in enumerate(images, start=1):
                        img_path: str = record.imgpath
                        timestamp: str = str(record.timestamp)
                        filename: str = Path(img_path).name
                        seq: int = total - rank + 1

                        with ui.element("div").classes("image-tile relative"):
                            # Delete overlay button
                            def delete_handler(target_id=int(record.id)) -> None:
                                result = engine._backend.delete_image(target_id)
                                if result.ok:
                                    ui.notify("Image deleted", type="positive")
                                    ui.navigate.to("/images")
                                else:
                                    ui.notify(result.err, type="negative")

                            ui.button(
                                icon="delete", color="red", on_click=delete_handler
                            ).props("round flat size=sm").classes(
                                "absolute top-2 right-2 bg-white/80 hover:bg-white text-red-500 z-10"
                            )

                            ui.image(img_path).style(
                                "width: 100%; height: 180px; object-fit: cover;"
                            )
                            with ui.element("div").classes("image-tile-meta"):
                                ui.label(f"#{seq} · Memory {rank}").classes(
                                    "image-tile-seq"
                                )
                                ui.label(filename).classes("image-tile-name")
                                ui.label(timestamp).classes("image-tile-ts")

    def run(self, title: str = "Saarthi", port: int = 8080) -> None:
        """Start the NiceGUI server.  This call blocks."""
        assert isinstance(title, str), "title must be a string"
        assert isinstance(port, int), "port must be an integer"
        logger.info("Starting Saarthi UI on port %d", port)
        ui.run(title=title, port=port, reload=False)


def _inject_css() -> None:
    """Inject shared CSS into the current page."""
    ui.add_head_html(_SHARED_CSS)


def _render_found_result(data: dict) -> None:
    """Render a successful FOUND search result inside the current UI context."""
    # Normalised query
    with ui.card().classes("w-full p-4 bg-gray-50 rounded-xl"):
        ui.label("Normalised Query").classes("text-sm font-medium text-gray-400 mb-1")
        ui.label(data["normalized_query"]).classes(
            "text-lg font-semibold text-gray-800"
        )

    # Score metadata chips
    with ui.row().classes("flex-wrap gap-3"):
        ui.label(f"Score: {data['final_score']:.2f}").classes("metadata-chip")
        ui.label(f"Semantic: {data['semantic_score']:.2f}").classes("metadata-chip")
        ui.label(f"Grounding: {data['grounding_score']:.2f}").classes("metadata-chip")
        ui.label(f"Candidates: {data['candidate_count']}").classes("metadata-chip")
        ui.label(f"Captured: {data['image_timestamp']}").classes("metadata-chip")
        ui.label(
            f"SLM: {data['slm_duration_s']}s | Search: {data['search_duration_s']}s"
        ).classes("metadata-chip")

    # Original image — bounding box not required in user result (spec §18)
    ui.image(data["image_path"]).classes("result-image mx-auto mt-2")


def _render_not_found_result(data: dict) -> None:
    """Render a NOT_FOUND search result inside the current UI context."""
    with ui.card().classes("w-full p-4 rounded-xl"):
        ui.label("Object Not Found").classes("text-xl font-bold status-error mb-2")
        ui.label(f"Query: {data['normalized_query']}").classes(
            "text-gray-700 font-medium mb-1"
        )
        ui.label(data["reason"]).classes("text-gray-500 text-sm mb-2")
        if data.get("candidate_count", 0) > 0:
            best: str = (
                f"  Best score: {data['best_score']:.2f}"
                if "best_score" in data
                else ""
            )
            ui.label(
                f"{data['candidate_count']} candidate(s) evaluated.{best}"
            ).classes("metadata-chip")
