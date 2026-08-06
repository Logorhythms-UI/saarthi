import logging
import os

from nicegui import app, events, ui

from ui.mock_model import process_image

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)

# Ensure storage directory exists
STORAGE_DIR: str = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(STORAGE_DIR, exist_ok=True)

# Add static files route for displaying images
app.add_static_files("/uploads", STORAGE_DIR)

# State to keep track of uploaded files
uploaded_image_paths: list[str] = []

# Global references for UI components
upload_status_label: ui.label | None = None
describe_section: ui.card | None = None
results_section: ui.card | None = None
results_container: ui.column | None = None


async def handle_upload(e: events.UploadEventArguments) -> None:
    """Handles the file upload event and saves the image to the storage directory."""
    assert e is not None, "Upload event arguments cannot be None"
    assert e.file is not None, "Uploaded file must exist"
    assert e.file.name is not None, "Uploaded file must have a name"

    logger.info(f"Uploading file: {e.file.name}")
    try:
        file_path: str = os.path.join(STORAGE_DIR, e.file.name)
        await e.file.save(file_path)
        uploaded_image_paths.append(file_path)
        logger.info(f"File saved to: {file_path}")
        ui.notify(f"Uploaded {e.file.name}", type="positive")
        update_ui_state()
    except Exception as ex:  # noqa: BLE001
        logger.error(f"Failed to upload file {e.file.name}: {ex}")
        ui.notify(f"Failed to upload {e.file.name}", type="negative")


def process_images(prompt: str) -> None:
    """Processes all uploaded images using the vision model and displays the results."""
    assert isinstance(prompt, str), "Prompt must be a string"

    if not prompt:
        ui.notify("Please enter a description of the object to find.", type="warning")
        return

    if not uploaded_image_paths:
        ui.notify("Please upload at least one image first.", type="warning")
        return

    logger.info(
        f"Processing {len(uploaded_image_paths)} images with prompt: '{prompt}'"
    )

    if results_section is not None:
        results_section.set_visibility(True)

    if results_container is None:
        return

    results_container.clear()

    with results_container:
        for img_path in uploaded_image_paths:
            try:
                result_path: str | None = process_image(img_path, prompt)
                if result_path is not None:
                    assert isinstance(result_path, str), "Result path must be a string"
                    filename: str = os.path.basename(result_path)
                    image_url: str = f"/uploads/{filename}"

                    with ui.card().classes(
                        "w-full items-center no-shadow border border-gray-200"
                    ):
                        ui.label(f"Processed: {filename}").classes(
                            "text-md font-semibold text-gray-700"
                        )
                        ui.image(image_url).classes(
                            "w-full max-w-lg rounded-lg shadow-sm mt-4"
                        )
                else:
                    ui.notify(f"Processing failed for {img_path}", type="negative")
            except Exception as ex:  # noqa: BLE001
                logger.error(f"Error processing image {img_path}: {ex}")
                ui.notify("Error processing image", type="negative")

    ui.notify("Processing complete!", type="positive")


def update_ui_state() -> None:
    """Updates UI elements based on current state (e.g., number of uploaded files)."""
    if upload_status_label:
        upload_status_label.set_text(f"{len(uploaded_image_paths)} image(s) uploaded.")

    if describe_section and len(uploaded_image_paths) > 0:
        describe_section.set_visibility(True)


@ui.page("/")
def main_page() -> None:
    """Main entry point for the UI application."""
    logger.info("Rendering main page.")

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
    </style>
    """)

    with ui.column().classes("w-full max-w-4xl mx-auto p-6 items-stretch"):
        ui.label("Saarthi").classes(
            "text-5xl font-extrabold mb-8 text-center gradient-text tracking-tight"
        )

        # 1. Upload Section (Always visible)
        with ui.card().classes("glass-card w-full"):
            ui.label("Upload Images").classes("text-2xl font-bold mb-4 text-gray-800")
            ui.upload(multiple=True, on_upload=handle_upload, auto_upload=True).classes(
                "w-full border-2 border-dashed border-gray-300 rounded-xl p-4 bg-gray-50/50 hover:bg-gray-50 transition-colors"
            )

            global upload_status_label
            upload_status_label = ui.label("0 image(s) uploaded.").classes(
                "mt-4 text-gray-500 font-medium"
            )

        # 2. Describe Section (Initially hidden)
        global describe_section
        with ui.card().classes("glass-card w-full") as describe_section:
            describe_section.set_visibility(False)
            ui.label("Describe Object").classes("text-2xl font-bold mb-4 text-gray-800")
            prompt_input = ui.input(
                label="What should the model find?", placeholder="e.g., a red apple"
            ).classes("w-full text-lg mb-6")

            ui.button(
                "Process Images",
                on_click=lambda: process_images(prompt_input.value or ""),
            ).classes("primary-btn w-full text-lg")

        # 3. Results Section (Initially hidden)
        global results_section, results_container
        with ui.card().classes("glass-card w-full") as results_section:
            results_section.set_visibility(False)
            ui.label("Results").classes("text-2xl font-bold mb-6 text-gray-800")
            results_container = ui.column().classes("w-full items-stretch gap-6")


if __name__ in {"__main__", "__mp_main__"}:
    logger.info("Starting Saarthi UI application")
    ui.run(title="Saarthi", port=8080)
