# black formatting is not done yet

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
upload_status_label: ui.label | None = None
results_container: ui.column | None = None


def handle_upload(e: events.UploadEventArguments) -> None:
    """Handles the file upload event and saves the image to the storage directory."""
    assert e is not None, "Upload event arguments cannot be None"
    assert e.name is not None, "Uploaded file must have a name"
    assert e.content is not None, "Uploaded file must have content"

    logger.info(f"Uploading file: {e.name}")
    try:
        file_path: str = os.path.join(STORAGE_DIR, e.name)
        with open(file_path, "wb") as f:
            f.write(e.content.read())
        uploaded_image_paths.append(file_path)
        logger.info(f"File saved to: {file_path}")
        ui.notify(f"Uploaded {e.name}", type="positive")
        update_ui_state()
    except Exception as ex:  # noqa: BLE001
        logger.error(f"Failed to upload file {e.name}: {ex}")
        ui.notify(f"Failed to upload {e.name}", type="negative")


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

    # Clear previous results container
    if not results_container:
        return

    results_container.clear()

    with results_container:
        for img_path in uploaded_image_paths:
            try:
                result_path: str | None = process_image(img_path, prompt)
                if result_path is not None:
                    assert isinstance(result_path, str), "Result path must be a string"
                    # The static route serves from /uploads/...
                    # Assuming result_path is like 'ui/uploads/image.jpg'
                    # We need to construct the URL for nicegui
                    filename: str = os.path.basename(result_path)
                    image_url: str = f"/uploads/{filename}"

                    with ui.card().classes("w-full items-center"):
                        ui.label(f"Processed: {filename}").classes("text-lg font-bold")
                        ui.image(image_url).classes("w-full max-w-lg rounded")
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


@ui.page("/")
def main_page() -> None:
    """Main entry point for the UI application."""
    logger.info("Rendering main page.")

    with ui.column().classes("w-full max-w-3xl mx-auto p-4 items-stretch"):
        ui.label("Saarthi Vision Model UI").classes(
            "text-3xl font-bold mb-6 text-center"
        )

        with ui.card().classes("w-full mb-6"):
            ui.label("1. Upload Images").classes("text-xl font-semibold mb-2")
            ui.upload(multiple=True, on_upload=handle_upload, auto_upload=True).classes(
                "w-full"
            )
            global upload_status_label
            upload_status_label = ui.label("0 image(s) uploaded.").classes(
                "mt-2 text-gray-600"
            )

        with ui.card().classes("w-full mb-6"):
            ui.label("2. Describe Object").classes("text-xl font-semibold mb-2")
            prompt_input = ui.input(
                label="What should the model find?", placeholder="e.g., a red apple"
            ).classes("w-full")

            ui.button(
                "Process Images",
                on_click=lambda: process_images(prompt_input.value or ""),
            ).classes("w-full mt-4")

        ui.label("3. Results").classes("text-xl font-semibold mb-2 mt-4")
        global results_container
        results_container = ui.column().classes("w-full items-center gap-4")


if __name__ in {"__main__", "__mp_main__"}:
    logger.info("Starting Saarthi UI application")
    ui.run(title="Saarthi", port=8080)
