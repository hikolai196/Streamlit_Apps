# Meeting Transcription App

A local MVP meeting transcription app built with **Python** and **Streamlit**. Upload audio files, transcribe with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), view timestamped segments, optionally convert Chinese output to Traditional Chinese, and export to TXT, SRT, or DOCX.

## Features

- Upload `.wav`, `.mp3`, or `.m4a` audio files
- Language settings: Auto / Chinese / English
- Whisper models: `small`, `medium`, `large-v3`
- Audio preprocessing: mono, 16 kHz WAV, light normalization
- Segment-level transcript with timestamps
- Optional Simplified → Traditional Chinese conversion (OpenCC)
- Export: TXT, SRT, DOCX
- **Batch file processing** with ZIP download of all exports
- **Progress bar with estimated time remaining**
- Runs fully locally — no cloud, auth, or database

## Requirements

- **Python 3.10+**
- **ffmpeg** (system dependency for MP3/M4A decoding; `imageio-ffmpeg` is also bundled as a fallback)

### Installing ffmpeg

| OS | Command |
|----|---------|
| Windows (winget) | `winget install Gyan.FFmpeg` |
| Windows (chocolatey) | `choco install ffmpeg` |
| macOS | `brew install ffmpeg` |
| Ubuntu/Debian | `sudo apt install ffmpeg` |

Verify installation:

```bash
ffmpeg -version
```

## Setup

```bash
cd transcription_app
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install pytest   # optional, for running tests
```

On first transcription, faster-whisper downloads the selected model automatically.

## Run

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Usage

1. Choose **Single file** or **Batch files** in the sidebar.
2. Upload one or more audio files (`.wav`, `.mp3`, or `.m4a`).
3. Review file metadata (batch mode shows a queue table).
4. Adjust settings in the sidebar:
   - **Language**: Auto-detect, Chinese, or English
   - **Model**: `small` (fast), `medium` (balanced), `large-v3` (most accurate)
   - **Include timestamps**: prefix lines in TXT/DOCX exports
   - **Convert to Traditional Chinese**: normalize Chinese characters via OpenCC
5. Click **Transcribe** (single) or **Transcribe N files** (batch).
6. Watch the progress bar and estimated time remaining.
7. View transcripts and download TXT, SRT, DOCX, or a batch ZIP.

## Project Structure

```
transcription_app/
├── app.py                 # Streamlit UI
├── config.py              # Constants and paths
├── requirements.txt
├── modules/
│   ├── upload.py          # File validation and save
│   ├── preprocess.py      # Audio conversion
│   ├── asr.py             # faster-whisper transcription
│   ├── pipeline.py        # Single/batch pipeline with progress stages
│   ├── postprocess.py     # Text cleanup and OpenCC
│   ├── export_txt.py
│   ├── export_srt.py
│   └── export_docx.py
├── utils/
│   ├── file_utils.py
│   ├── time_utils.py
│   └── logger.py
├── temp/                  # Temporary uploaded/processed files
├── output/                # Generated export files
└── tests/
```

## Tests

```bash
cd transcription_app
pytest tests/ -v
```

## Limitations (MVP)

- No real-time streaming transcription
- No speaker diarization (who said what)
- No video upload support
- No summary or AI post-processing
- Model download and first run can be slow on CPU
- Very long files may take significant time and memory
- Mixed dialects and heavy background noise reduce accuracy

## Future Improvements

- Custom vocabulary / hotwords
- GPU detection and settings in UI
- Speaker diarization (e.g. pyannote)
- Punctuation and paragraph grouping
- VTT export format
- Docker packaging for easier deployment

## License

MIT (or your preferred license)
