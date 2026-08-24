import logging

from backend.api import backend_api
from ui.main import frontend_engine

logging.basicConfig(
    level=logging.INFO, format="MAIN - %(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the Saarthi application.

    Initialises the backend (which loads DB, storage, and AI model),
    wires it into the frontend, and starts the NiceGUI server.
    """
    logger.info("Initialising Saarthi...")

    backend: backend_api = backend_api()
    frontend: frontend_engine = frontend_engine(backend)
    frontend.setup_routes()

    logger.info("All systems ready — launching UI")
    frontend.run()


if __name__ == "__main__":
    main()
