import logging

from nicegui import app, events, ui

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class frontend_engine():
    def __init__(self):
        self.logger: logging.Logger = logging.getLogger(__name__)   
            


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
            ui.upload(multiple=True, on_upload=lambda: None, auto_upload=True).classes(
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

            ui.button("Process Images", on_click=lambda: None).classes(
                "primary-btn w-full text-lg"
            )

        # 3. Results Section (Initially hidden)
        global results_section, results_container
        with ui.card().classes("glass-card w-full") as results_section:
            results_section.set_visibility(False)
            ui.label("Results").classes("text-2xl font-bold mb-6 text-gray-800")
            results_container = ui.column().classes("w-full items-stretch gap-6")


if __name__ in {"__main__", "__mp_main__"}:
    logger.info("Starting Saarthi UI application")
    ui.run(title="Saarthi", port=8080)
