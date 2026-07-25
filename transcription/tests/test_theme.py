"""Tests for UI theme helpers."""

from utils.theme import (
    DEFAULT_THEME,
    apply_streamlit_theme,
    normalize_theme,
    theme_config_options,
)


def test_default_theme_is_dark():
    assert DEFAULT_THEME == "dark"
    assert normalize_theme(None) == "dark"
    assert normalize_theme("nope") == "dark"


def test_theme_config_options_include_base():
    dark = theme_config_options("dark")
    light = theme_config_options("light")
    assert dark["theme.base"] == "dark"
    assert light["theme.base"] == "light"
    assert dark["theme.backgroundColor"] != light["theme.backgroundColor"]


def test_apply_streamlit_theme_sets_options():
    applied: dict[str, str] = {}

    def set_option(key: str, value: str) -> None:
        applied[key] = value

    resolved = apply_streamlit_theme("light", set_option=set_option)
    assert resolved == "light"
    assert applied["theme.base"] == "light"
    assert "theme.primaryColor" in applied
