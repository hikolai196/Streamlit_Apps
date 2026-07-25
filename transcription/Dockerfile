# Meeting Transcription App — CPU image with system ffmpeg
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Persist Whisper / Hugging Face downloads across restarts via volume
    HF_HOME=/cache/huggingface \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock requirements.txt ./
# Prefer frozen runtime pins; requirements.txt also lists pytest for local/CI.
RUN pip install --upgrade pip \
    && pip install -r requirements.lock

COPY app.py config.py pyproject.toml ./
COPY modules ./modules
COPY ui ./ui
COPY utils ./utils
COPY .streamlit ./.streamlit

RUN mkdir -p /app/temp /app/output /cache/huggingface \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app /cache

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
