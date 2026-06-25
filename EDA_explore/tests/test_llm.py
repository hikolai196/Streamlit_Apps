from unittest.mock import patch

from eda_agent.llm.client import MODEL, SYSTEM_PROMPT, generate_code


@patch("eda_agent.llm.client.ollama.chat")
def test_generate_code_calls_ollama(mock_chat):
    mock_chat.return_value = {"message": {"content": 'result = df["sales"].mean()'}}

    code = generate_code("What is average sales?")

    assert code == 'result = df["sales"].mean()'
    mock_chat.assert_called_once()

    call_kwargs = mock_chat.call_args.kwargs
    assert call_kwargs["model"] == MODEL
    assert call_kwargs["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert call_kwargs["messages"][-1] == {
        "role": "user",
        "content": "What is average sales?",
    }


@patch("eda_agent.llm.client.ollama.chat")
def test_generate_code_includes_recent_history(mock_chat):
    mock_chat.return_value = {"message": {"content": "result = 1"}}

    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]

    generate_code("latest question", history)

    messages = mock_chat.call_args.kwargs["messages"]
    history_messages = messages[1:-1]

    assert len(history_messages) == 4
    assert history_messages[-1]["content"] == "a3"


@patch("eda_agent.llm.client.ollama.chat")
def test_generate_code_without_history(mock_chat):
    mock_chat.return_value = {"message": {"content": "result = 0"}}

    generate_code("hello")

    messages = mock_chat.call_args.kwargs["messages"]
    assert len(messages) == 2
