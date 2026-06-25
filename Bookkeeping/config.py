# config.py
# Loads .env, exposes all constants to the app
# 
# Single source of truth for all application constants.
# Loads values from .env (via python-dotenv) so nothing is hardcoded here.
#
# All other modules import from this file — nothing else should touch .env directly.

import os
from dotenv import load_dotenv

# Resolve .env next to this file, regardless of cwd
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(_ENV_PATH)

# ---------------------------------------------------------------------------
# Helper: read a required env var, raise clearly if missing
# ---------------------------------------------------------------------------

def _require(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise EnvironmentError(
            f"Required environment variable '{key}' is missing from .env ({_ENV_PATH})"
        )
    return value

def _csv_list(key: str) -> list[str]:
    """Read a comma-separated env var into a list, stripping whitespace."""
    return [v.strip() for v in _require(key).split(',') if v.strip()]

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE  = os.path.join(_BASE_DIR, _require('CSV_FILE_NAME'))

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

DETAILS_MAX_LENGTH: int = int(_require('DETAILS_MAX_LENGTH'))

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

APP_TITLE   = _require('APP_TITLE')
PAGE_LAYOUT = _require('PAGE_LAYOUT')

# ---------------------------------------------------------------------------
# Categories & payments
# ---------------------------------------------------------------------------

CATEGORY_LIST:    list[str] = _csv_list('CATEGORY_LIST')
INCOME_CATEGORIES: set[str] = set(_csv_list('INCOME_CATEGORIES'))
PAYMENT_LIST:     list[str] = _csv_list('PAYMENT_LIST')

# ---------------------------------------------------------------------------
# Color maps — built dynamically from .env so adding a category only requires
# updating .env, not touching Python source.
# ---------------------------------------------------------------------------

# Category colors  (COLOR_<category_name>=<hex>)
CATEGORY_COLORS: dict[str, str] = {
    cat: _require(f'COLOR_{cat}')
    for cat in CATEGORY_LIST
}

# Income/Expense type colors  (COLOR_TYPE_Income, COLOR_TYPE_Expense)
TYPE_COLORS: dict[str, str] = {
    'Income':  _require('COLOR_TYPE_Income'),
    'Expense': _require('COLOR_TYPE_Expense'),
}

# Payment method colors  (COLOR_PAY_<payment_name>=<hex>)
PAYMENT_COLORS: dict[str, str] = {
    pay: _require(f'COLOR_PAY_{pay}')
    for pay in PAYMENT_LIST
}