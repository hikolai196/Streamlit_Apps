"""Application configuration and constants."""

from __future__ import annotations

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "output"

# Supported media formats (audio + video containers)
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov"}
ALLOWED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS
ALLOWED_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "video/mp4",
    "video/x-matroska",
    "video/webm",
    "video/quicktime",
    "application/octet-stream",
}

# Upload limits (reject before / after save)
MAX_UPLOAD_SIZE_MB = 500
MAX_DURATION_SEC = 3 * 60 * 60  # 3 hours

# ASR settings
WHISPER_MODELS = ["small", "medium", "large-v3"]
DEFAULT_MODEL = "medium"
DEFAULT_LANGUAGE_LABEL = "Auto"
DEFAULT_DEVICE = "auto"
DEVICE_OPTIONS = ["auto", "cpu", "cuda"]

LANGUAGE_OPTIONS = {
    "Auto": None,
    "Chinese": "zh",
    "English": "en",
}

# Post-process: start a new paragraph when silence between segments exceeds this
DEFAULT_PARAGRAPH_GAP_SEC = 1.5

# Long-audio chunking (split → transcribe → merge)
CHUNK_ENABLED_DEFAULT = True
CHUNK_DURATION_SEC = 10 * 60  # 10 minutes
CHUNK_OVERLAP_SEC = 5.0
CHUNK_MIN_DURATION_SEC = 12 * 60  # only chunk when longer than this

# Batch parallelism (files processed per Streamlit step)
MAX_PARALLEL_JOBS_DEFAULT = 1
MAX_PARALLEL_JOBS_OPTIONS = [1, 2, 3, 4]

# Simple gap-based diarization: new speaker after this silence
DIARIZATION_GAP_SEC = 2.0

# Progress: treat each minute of audio as one batch work unit (min 1.0)
PROGRESS_SECONDS_PER_UNIT = 60.0
# Early ETA fallback: assume transcription runs at this fraction of realtime
PROGRESS_REALTIME_FACTOR = 0.35

# Audio preprocessing
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

# OpenCC conversion profile (Simplified -> Traditional)
OPENCC_CONFIG = "s2t"

# Optional app password (set env TRANSCRIPTION_APP_PASSWORD to enable login gate)
APP_PASSWORD = os.environ.get("TRANSCRIPTION_APP_PASSWORD", "").strip()
