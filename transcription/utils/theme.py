"""Light/dark theme helpers for the Streamlit UI."""

from __future__ import annotations

from typing import Any

# Per-mode Streamlit theme.* options (teal accent kept from app branding).
THEME_OPTIONS: dict[str, dict[str, str]] = {
    "dark": {
        "theme.base": "dark",
        "theme.primaryColor": "#2DD4BF",
        "theme.backgroundColor": "#0B1220",
        "theme.secondaryBackgroundColor": "#1E293B",
        "theme.textColor": "#E2E8F0",
    },
    "light": {
        "theme.base": "light",
        "theme.primaryColor": "#0F766E",
        "theme.backgroundColor": "#F8FAFC",
        "theme.secondaryBackgroundColor": "#EEF2F7",
        "theme.textColor": "#0F172A",
    },
}

DEFAULT_THEME = "dark"


def normalize_theme(theme: str | None) -> str:
    """Return a valid theme key; default to dark."""
    if theme in THEME_OPTIONS:
        return theme
    return DEFAULT_THEME


def theme_config_options(theme: str) -> dict[str, str]:
    """Return Streamlit config option map for the given theme."""
    return dict(THEME_OPTIONS[normalize_theme(theme)])


def apply_streamlit_theme(theme: str, set_option: Any | None = None) -> str:
    """
    Apply theme options via Streamlit's config API.

    ``set_option`` defaults to ``st._config.set_option`` so tests can inject a stub.
    """
    if set_option is None:
        import streamlit as st

        set_option = st._config.set_option

    resolved = normalize_theme(theme)
    for key, value in theme_config_options(resolved).items():
        set_option(key, value)
    return resolved
