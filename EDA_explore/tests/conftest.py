import io

import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    return pd.DataFrame({"sales": [10, 20, 30], "region": ["A", "B", "A"]})


@pytest.fixture
def csv_upload():
    buffer = io.BytesIO(b"sales,region\n10,A\n20,B\n30,A\n")
    buffer.name = "data.csv"
    return buffer


@pytest.fixture
def session_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("eda_agent.memory.sessions.SESSION_DIR", tmp_path)
    return tmp_path
