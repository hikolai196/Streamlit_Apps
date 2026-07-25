# Meeting Transcription App

Local meeting transcription with **Python**, **Streamlit**, and [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Upload audio or video, transcribe on your machine, edit the transcript, and export TXT / SRT / VTT / DOCX (or a batch ZIP).

Version: **0.3.0** (see `pyproject.toml`)

## Features

- Audio (`.wav`, `.mp3`, `.m4a`) and video (`.mp4`, `.mkv`, `.webm`, `.mov`) upload
- Languages: Auto / Chinese / English; models: `small`, `medium`, `large-v3`
- Device: auto / CPU / CUDA (falls back to CPU when GPU is unavailable)
- Preprocess to mono 16 kHz WAV; optional long-audio chunking
- Hotwords, initial prompt, paragraph grouping by silence
- Optional gap-based speaker labels and extractive summary / action items
- Editable transcript, audio player with seek-to-segment
- Single-file and batch modes (filters, retry, cancel between batches, 1–4 parallel workers)
- Progress bar with duration-aware ETA
- Dark theme by default; sidebar light/dark toggle
- Upload limits: 500 MB / 3 hours (configurable in `config.py`)
- Optional password gate: `TRANSCRIPTION_APP_PASSWORD`
- Docker image + Compose with healthcheck and persistent model cache

## Requirements

- **Python 3.10–3.12** (recommended; 3.13+ may break until `pydub` / `audioread` catch up)
- **ffmpeg** (system install preferred; `imageio-ffmpeg` is a fallback)

| OS | Install ffmpeg |
|----|----------------|
| Windows (winget) | `winget install Gyan.FFmpeg` |
| Windows (chocolatey) | `choco install ffmpeg` |
| macOS | `brew install ffmpeg` |
| Ubuntu/Debian | `sudo apt install ffmpeg` |

## Setup

```bash
cd transcription_app
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Runtime + test tools (single requirements file)
pip install -r requirements.txt

# Optional: exact frozen runtime pins (Docker / reproducible installs)
pip install -r requirements.lock

# Optional GPU (install the torch wheel that matches your CUDA toolkit)
# https://pytorch.org/get-started/locally/
pip install torch
```

On first transcription, faster-whisper downloads the selected model into the HF cache.

## Run (local)

```bash
streamlit run app.py
```

Open the URL in the terminal (usually `http://localhost:8501`). Settings live in `.streamlit/config.toml` (localhost bind, 500 MB upload cap, dark theme). Use the sidebar **Dark theme** toggle to switch.

## Docker

```bash
cp .env.example .env   # optional password / host port
docker compose up --build
```

Then open `http://localhost:8501` (or `HOST_PORT` from `.env`).

| Setting | Purpose |
|---------|---------|
| `./temp` → `/app/temp` | Uploads / intermediate audio |
| `./output` → `/app/output` | Exports |
| volume `whisper-cache` | Whisper / Hugging Face model cache |
| `TRANSCRIPTION_APP_PASSWORD` | Optional login gate |
| `GET /_stcore/health` | Docker / CI healthcheck |

Native runs bind to **localhost**. The container binds `0.0.0.0` for the published port — use a reverse proxy + HTTPS if exposing beyond a trusted network.

**RAM guidance:** `small` ≈ 2–4 GB; `medium` ≈ 4–8 GB; `large-v3` needs more (and a large first download).

```bash
python scripts/smoke_check.py
python scripts/smoke_check.py --http   # when the app is already up
```

## Usage

1. Choose **Single file** or **Batch files** in the sidebar.
2. Upload media (audio and/or video as allowed).
3. Adjust language, model, device, timestamps, Traditional Chinese, paragraphs, hotwords/prompt, chunking, speaker labels, summary, and batch parallelism.
4. Click **Transcribe** (or **Transcribe N files**).
5. Review / edit the transcript, seek audio, download exports or a batch ZIP.
   - Empty (no speech) batch items skip export files.

## Project structure

```
transcription_app/
├── app.py                 # Streamlit entrypoint
├── config.py              # Paths, limits, ASR / chunk / auth settings
├── requirements.txt       # Runtime + pytest/pytest-cov
├── requirements.lock      # Frozen runtime pins (Docker)
├── pyproject.toml         # Package metadata, pytest, coverage
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── LICENSE
├── RELEASE_CHECKLIST.md
├── scripts/smoke_check.py
├── .streamlit/config.toml
├── .github/workflows/ci.yml
├── modules/               # Pipeline, ASR, media, exports, summary, …
├── ui/                    # Transcript, batch, Streamlit components
├── utils/                 # Files, auth, theme, progress, ffmpeg helper
├── temp/                  # Runtime uploads (gitignored contents)
├── output/                # Runtime exports (gitignored contents)
└── tests/
```

## Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
pytest tests/ -v --cov=modules --cov=utils --cov=config --cov=ui --cov-fail-under=70
python scripts/smoke_check.py
```

CI runs the suite on Python 3.10 and 3.12, then builds the Docker image and waits for `/_stcore/health`.

## Limitations

- No real-time streaming transcription
- Speaker labels and summary are heuristics (not pyannote / LLM)
- Mid-file cancel is limited by Streamlit’s synchronous model
- Long files and large models need substantial CPU/RAM (or CUDA)

## Security

- Default bind: `localhost` (see `.streamlit/config.toml`)
- Optional gate: `TRANSCRIPTION_APP_PASSWORD`
- Do not expose publicly without reverse proxy + HTTPS

## License

MIT — see `LICENSE`.
