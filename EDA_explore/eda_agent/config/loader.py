from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.md"


def load_config(path=None):
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    config = {}
    current_key = None
    buffer = []

    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            if line.startswith("# "):
                if current_key:
                    config[current_key] = "\n".join(buffer).strip()
                    buffer = []

                current_key = line.replace("# ", "").strip()
            else:
                buffer.append(line)

        if current_key:
            config[current_key] = "\n".join(buffer).strip()

    return config
