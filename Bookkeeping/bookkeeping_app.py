"""
bookkeeping_app.py
------
Entry point for the Bookkeeping App.

Launch with:
    uv run streamlit run bookkeeping_app.py
    streamlit run bookkeeping_app.py

This file is intentionally thin — all logic lives in:
    config.py  — constants and environment loading
    data.py    — data processing and persistence
    charts.py  — Plotly figure builders
    ui.py      — Streamlit rendering
"""

from ui import main

if __name__ == "__main__":
    main()