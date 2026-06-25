import ollama

from eda_agent.config.loader import load_config

config = load_config()

MODEL = config.get("MODEL", "gemma:2b")
SYSTEM_PROMPT = config.get("SYSTEM_PROMPT", "")


def generate_code(user_input, history=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history[-4:])

    messages.append({"role": "user", "content": user_input})

    response = ollama.chat(
        model=MODEL,
        messages=messages,
    )

    return response["message"]["content"]
