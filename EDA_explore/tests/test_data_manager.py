import io

import pandas as pd

from eda_agent.data.manager import DataManager


def test_load_csv(csv_upload):
    manager = DataManager()
    manager.load_file(csv_upload)

    assert list(manager.df.columns) == ["sales", "region"]
    assert len(manager.df) == 3


def test_get_preview_returns_head(sample_df):
    manager = DataManager()
    manager.df = sample_df

    preview = manager.get_preview()

    assert len(preview) == 3
    assert list(preview.columns) == ["sales", "region"]


def test_get_schema(sample_df):
    manager = DataManager()
    manager.df = sample_df

    schema = manager.get_schema()

    assert "sales" in schema
    assert "region" in schema


def test_load_excel():
    df = pd.DataFrame({"value": [1, 2]})
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    buffer.name = "data.xlsx"

    manager = DataManager()
    manager.load_file(buffer)

    assert list(manager.df.columns) == ["value"]
    assert len(manager.df) == 2
