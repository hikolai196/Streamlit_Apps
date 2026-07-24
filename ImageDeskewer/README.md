# Image Deskewer (`dimage`)

Interactive Streamlit app that perspective-corrects (deskews) photos by selecting four corners.

**Python 3.11+** · dependencies live in `pyproject.toml` only.

## Features

- Upload PNG/JPG, pick corners (click, auto-detect, or keyboard coordinates)
- Auto-order corners, side-by-side preview, post-process (rotate / crop / contrast / sharpen)
- Session history, dark/light toggle, privacy-first (in-session images; usage stats off)

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

Run the app:

```bash
streamlit run main.py
```

Quality checks:

```bash
pytest
ruff check dimage tests main.py
mypy dimage
```

## Usage

1. Upload an image (max 20 MB, max 10000px per side).
2. Mark four corners (any order), or use **Auto-detect** / coordinate fields.
3. Adjust post-process options in the sidebar.
4. Download, or **Save to history** for this session.

## Deploy

### Docker

```bash
docker build -t dimage .
docker run --rm -p 8501:8501 dimage
```

Open http://localhost:8501.

### Streamlit Community Cloud

1. Push to GitHub and create an app at [share.streamlit.io](https://share.streamlit.io).
2. Main file: `main.py`.
3. Dependencies are declared in `pyproject.toml`. Prefer Docker deploy if the Cloud installer does not pick them up automatically.

### Cloud Run

Build/push the Docker image and expose port `8501` (`0.0.0.0` is already set in the Dockerfile).

## Layout

```
.
├── dimage/                 # App package
│   ├── processing.py       # Image load / deskew / post-process
│   └── ui.py               # Streamlit UI
├── main.py                 # streamlit run entry
├── tests/
├── .streamlit/config.toml
├── .github/workflows/ci.yml
├── Dockerfile
├── pyproject.toml          # Deps, Python pin, tool config
├── LICENSE
└── README.md
```

## Troubleshooting

- **Bad image / deskew error:** use a valid PNG/JPG and a non-degenerate quadrilateral.
- **Install / start issues:** `pip install -e ".[dev]"` then `streamlit run main.py`.
- **Canvas quirks:** `streamlit-drawable-canvas` is lightly maintained; pin Streamlit if upgrades break it.

## License

MIT — see [LICENSE](LICENSE).
