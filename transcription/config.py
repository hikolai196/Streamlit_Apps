"""Application configuration and constants."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "output"

# Supported audio formats
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a"}
ALLOWED_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "application/octet-stream",
}

# ASR settings
WHISPER_MODELS = ["small", "medium", "large-v3"]
DEFAULT_MODEL = "medium"
DEFAULT_LANGUAGE = "auto"

LANGUAGE_OPTIONS = {
    "Auto": None,
    "Chinese": "zh",
    "English": "en",
}

# Audio preprocessing
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

# OpenCC conversion profile (Simplified -> Traditional)
OPENCC_CONFIG = "s2t"
