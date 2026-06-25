from eda_agent.memory.sessions import list_sessions, load_session, save_session


def test_list_sessions_empty(session_dir):
    assert list_sessions() == []


def test_save_and_load_session(session_dir):
    messages = [
        {"role": "user", "content": "average sales"},
        {"role": "assistant", "content": "20.0"},
    ]

    save_session("demo", messages)

    assert list_sessions() == ["demo"]
    assert load_session("demo") == messages


def test_load_session_missing_returns_empty(session_dir):
    assert load_session("missing") == []
