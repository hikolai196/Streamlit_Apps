from eda_agent.config.loader import load_config


def test_load_config_parses_sections(tmp_path):
    config_file = tmp_path / "config.md"
    config_file.write_text(
        "# MODEL\ngemma:2b\n\n# SYSTEM_PROMPT\nYou are an analyst.\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config["MODEL"] == "gemma:2b"
    assert config["SYSTEM_PROMPT"] == "You are an analyst."


def test_load_default_config_from_project_root():
    config = load_config()

    assert "MODEL" in config
    assert "SYSTEM_PROMPT" in config
    assert config["MODEL"] == "gemma:2b"
