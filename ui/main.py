import logging
import tempfile
from pathlib import Path

from nicegui import ui
from nicegui.events import UploadEventArguments

from backend.api import backend_api
from utils.result import Result

logging.basicConfig(
    level=logging.INFO, format="UI - %(asctime)s - %(levelname)s - %(message)s"
)


class frontend_engine:
    """Provides the NiceGUI web dashboard for image upload and object search."""

    def __init__(self, backend: backend_api) -> None:
        assert isinstance(
            backend, backend_api
        ), "backend must be a backend_api instance"

        self.logger: logging.Logger = logging.getLogger(__name__)
        self.backend: backend_api = backend
        self.upload_count: int = 0
        self.logger.info("Frontend Engine Initialized")

    def setup_routes(self) -> None:
        """Register all NiceGUI page routes. Must be called before run()."""

        # Capture self for use inside the nested route function
        engine: frontend_engine = self

        @ui.page("/")
        def main_page() -> None:
            """Main dashboard with upload and search sections."""
            engine.logger.info("Rendering main page")

            # Inject premium CSS
            ui.add_head_html("""
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
            </style>
            """)

            with ui.column().classes("w-full max-w-4xl mx-auto p-6 items-stretch"):
                ui.label("Saarthi").classes(
                    "text-5xl font-extrabold mb-2 text-center gradient-text tracking-tight"
                )
                ui.label("Episodic Memory Aid").classes(
                    "text-lg mb-8 text-center text-gray-500 font-medium"
                )

                # ── Upload Section ──────────────────────────────────
                with ui.card().classes("glass-card w-full"):
                    ui.label("Upload Images").classes(
                        "text-2xl font-bold mb-4 text-gray-800"
                    )
                    ui.label(
                        "Upload a .zip archive of images or individual image files."
                    ).classes("text-gray-500 mb-4")

                    upload_status: ui.label = ui.label("0 image(s) ingested.").classes(
                        "mt-4 text-gray-500 font-medium status-msg status-info"
                    )

                    async def handle_upload(e: UploadEventArguments) -> None:
                        """Handle uploaded files — zip archives or individual images."""
                        filename: str = e.file.name
                        engine.logger.info(f"Received upload: {filename}")

                        # Save the uploaded content to a temporary file
                        suffix: str = Path(filename).suffix
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=suffix
                        ) as tmp:
                            tmp_path: str = tmp.name

                        await e.file.save(tmp_path)

                        if suffix.lower() == ".zip":
                            result: Result = engine.backend.ingest_zip(tmp_path)
                            if result.ok:
                                engine.upload_count += result.result
                                upload_status.set_text(
                                    f"{engine.upload_count} image(s) ingested."
                                )
                                upload_status.classes(
                                    remove="status-error status-info",
                                    add="status-success",
                                )
                                ui.notify(
                                    f"Extracted and ingested {result.result} images from zip",
                                    type="positive",
                                )
                            else:
                                upload_status.set_text(f"Zip error: {result.err}")
                                upload_status.classes(
                                    remove="status-success status-info",
                                    add="status-error",
                                )
                                ui.notify(
                                    f"Zip upload failed: {result.err}", type="negative"
                                )
                        else:
                            result = engine.backend.ingest_image(tmp_path)
                            if result.ok:
                                engine.upload_count += 1
                                upload_status.set_text(
                                    f"{engine.upload_count} image(s) ingested."
                                )
                                upload_status.classes(
                                    remove="status-error status-info",
                                    add="status-success",
                                )
                                ui.notify(f"Ingested {filename}", type="positive")
                            else:
                                upload_status.set_text(f"Upload error: {result.err}")
                                upload_status.classes(
                                    remove="status-success status-info",
                                    add="status-error",
                                )
                                ui.notify(
                                    f"Upload failed: {result.err}", type="negative"
                                )

                    ui.upload(
                        multiple=True,
                        on_upload=handle_upload,
                        auto_upload=True,
                    ).classes(
                        "w-full border-2 border-dashed border-gray-300 rounded-xl "
                        "p-4 bg-gray-50/50 hover:bg-gray-50 transition-colors"
                    ).props(
                        'accept=".zip,.jpg,.jpeg,.png,.bmp,.webp,.tiff"'
                    )

                # ── Search Section ──────────────────────────────────
                with ui.card().classes("glass-card w-full"):
                    ui.label("Search for an Object").classes(
                        "text-2xl font-bold mb-4 text-gray-800"
                    )
                    ui.label(
                        "Describe what you're looking for. The system will scan "
                        "your images from newest to oldest."
                    ).classes("text-gray-500 mb-4")

                    prompt_input: ui.input = ui.input(
                        label="What should the model find?",
                        placeholder="e.g., a red apple",
                    ).classes("w-full text-lg mb-6")

                    results_container: ui.column = ui.column().classes(
                        "w-full items-stretch gap-4"
                    )

                    def handle_search() -> None:
                        """Run reverse-chronological object detection search."""
                        query: str = prompt_input.value.strip()
                        if not query:
                            ui.notify("Please enter a search query.", type="warning")
                            return

                        engine.logger.info(f"Search requested: '{query}'")
                        results_container.clear()

                        with results_container:
                            ui.spinner("dots", size="lg").classes("mx-auto my-4")
                            ui.label(f"Searching for '{query}'...").classes(
                                "text-center text-gray-500 font-medium"
                            )

                        result: Result = engine.backend.search(query)

                        results_container.clear()
                        with results_container:
                            if result.ok:
                                annotated_path: str = result.result
                                ui.label(f"Found '{query}'!").classes(
                                    "text-xl font-bold text-center status-msg status-success"
                                )
                                ui.image(annotated_path).classes(
                                    "result-image mx-auto mt-4"
                                )
                                engine.logger.info(
                                    f"Search result displayed: {annotated_path}"
                                )
                            else:
                                ui.label(result.err).classes(
                                    "text-lg text-center status-msg status-error"
                                )
                                engine.logger.info(
                                    f"Search returned no results: {result.err}"
                                )

                    ui.button("Search", on_click=handle_search).classes(
                        "primary-btn w-full text-lg"
                    )

    def run(self, title: str = "Saarthi", port: int = 8080) -> None:
        """Start the NiceGUI server. This call blocks."""
        assert isinstance(title, str), "title must be a string"
        assert isinstance(port, int), "port must be an integer"

        self.logger.info(f"Starting Saarthi UI on port {port}")
        ui.run(title=title, port=port, reload=False)
