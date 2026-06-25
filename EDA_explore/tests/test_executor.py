import pandas as pd

from eda_agent.execution.executor import execute_code


def test_execute_code_returns_scalar(sample_df):
    code = 'result = df["sales"].mean()'
    result, error = execute_code(code, sample_df)

    assert error is None
    assert result == 20.0


def test_execute_code_returns_dataframe(sample_df):
    code = 'result = df.groupby("region")["sales"].sum()'
    result, error = execute_code(code, sample_df)

    assert error is None
    assert isinstance(result, pd.Series)
    assert result["A"] == 40


def test_execute_code_returns_none_when_no_result(sample_df):
    code = 'x = df["sales"].sum()'
    result, error = execute_code(code, sample_df)

    assert error is None
    assert result is None


def test_execute_code_captures_errors(sample_df):
    code = 'result = df["missing_column"].mean()'
    result, error = execute_code(code, sample_df)

    assert result is None
    assert error is not None
    assert "missing_column" in error
