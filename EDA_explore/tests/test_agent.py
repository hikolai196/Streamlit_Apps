from unittest.mock import patch

import pandas as pd

from eda_agent.agent import run_agent


@patch("eda_agent.agent.execute_code")
@patch("eda_agent.agent.generate_code")
def test_run_agent_success(mock_generate, mock_execute, sample_df):
    mock_generate.return_value = 'result = df["sales"].mean()'
    mock_execute.return_value = (20.0, None)

    code, result, error = run_agent("average sales", sample_df, [])

    assert code == 'result = df["sales"].mean()'
    assert result == 20.0
    assert error is None
    mock_generate.assert_called_once()
    mock_execute.assert_called_once_with('result = df["sales"].mean()', sample_df)


@patch("eda_agent.agent.execute_code")
@patch("eda_agent.agent.generate_code")
def test_run_agent_retries_on_error(mock_generate, mock_execute, sample_df):
    mock_generate.side_effect = [
        'result = df["bad"].mean()',
        'result = df["sales"].mean()',
    ]
    mock_execute.side_effect = [
        (None, "KeyError: bad"),
        (20.0, None),
    ]

    code, result, error = run_agent("average sales", sample_df, [])

    assert code == 'result = df["sales"].mean()'
    assert result == 20.0
    assert error is None
    assert mock_generate.call_count == 2
    assert mock_execute.call_count == 2

    fix_prompt = mock_generate.call_args_list[1].args[0]
    assert "KeyError: bad" in fix_prompt
    assert 'result = df["bad"].mean()' in fix_prompt


@patch("eda_agent.agent.execute_code")
@patch("eda_agent.agent.generate_code")
def test_run_agent_returns_error_after_retry(mock_generate, mock_execute, sample_df):
    mock_generate.side_effect = ["bad code", "still bad code"]
    mock_execute.side_effect = [
        (None, "first error"),
        (None, "second error"),
    ]

    code, result, error = run_agent("average sales", sample_df, [])

    assert code == "still bad code"
    assert result is None
    assert error == "second error"
