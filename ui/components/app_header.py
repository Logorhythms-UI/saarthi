"""Page header: product name, page title and a short description."""

from nicegui import ui


def app_header(product: str, title: str, subtitle: str) -> None:
    """Build the header. The title is the only h1 of the page."""
    assert product and title, "product and title are required"
    with ui.element("header").classes("up-header"):
        ui.label(product).classes("up-header__product")
        heading = ui.label(title).classes("up-header__title")
        heading.tag = "h1"
        ui.label(subtitle).classes("up-header__subtitle")
