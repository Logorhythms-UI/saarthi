import re
from pathlib import Path

from ui.theme import CSS_FILES, STATIC_DIR, stylesheet_links

COLOUR_PATTERN: re.Pattern[str] = re.compile(
    r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(", re.MULTILINE
)


def test_all_listed_css_files_exist() -> None:
    for relative_path in CSS_FILES:
        assert (STATIC_DIR / relative_path).is_file(), relative_path


def test_every_css_file_is_listed() -> None:
    on_disk: set[str] = {
        path.relative_to(STATIC_DIR).as_posix()
        for path in (STATIC_DIR / "css").rglob("*.css")
    }
    assert on_disk == set(CSS_FILES)


def test_only_tokens_file_has_colour_values() -> None:
    for relative_path in CSS_FILES:
        text: str = (STATIC_DIR / relative_path).read_text(encoding="utf-8")
        found: list[str] = COLOUR_PATTERN.findall(text)
        if relative_path == "css/tokens.css":
            assert found, "tokens.css must hold the colours"
        else:
            assert not found, f"{relative_path} has colour values: {found}"


def test_tokens_define_the_same_colours_for_light_and_dark() -> None:
    text: str = (STATIC_DIR / "css/tokens.css").read_text(encoding="utf-8")
    light, dark = text.split("body.body--dark")
    light_names: set[str] = set(re.findall(r"--(color-[a-z-]+):", light))
    dark_names: set[str] = set(re.findall(r"--(color-[a-z-]+):", dark))
    assert dark_names == light_names


def test_stylesheet_links_are_local() -> None:
    links: str = stylesheet_links()
    assert "http://" not in links and "https://" not in links
    assert links.count("<link") == len(CSS_FILES)
    assert Path(STATIC_DIR / "css/base.css").name in links
