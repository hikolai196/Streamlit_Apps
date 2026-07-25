# Release checklist

Before tagging a release or publishing an image.

## Version & pins

- [ ] Bump `version` in `pyproject.toml` when the change is user-visible
- [ ] Keep `requirements.txt` as the single day-to-day install (runtime + tests)
- [ ] Prefer `requirements.lock` for Docker / reproducible runtime installs
- [ ] Confirm `requires-python` matches CI (`3.10`–`3.12`)

## Quality gates

```bash
pip install -r requirements.txt
pytest tests/ -v --cov=modules --cov=utils --cov=config --cov=ui --cov-fail-under=70
python scripts/smoke_check.py
```

Optional (app already running):

```bash
python scripts/smoke_check.py --http
```

Optional manual check: upload a short clip and confirm TXT / SRT / VTT / DOCX downloads.

## Docker

```bash
docker compose build
docker compose up -d
python scripts/smoke_check.py --http
docker compose logs --tail=50 transcription
docker compose down
```

## Security / ops

- [ ] Native Streamlit still binds to `localhost`
- [ ] Container bind to `0.0.0.0` is intentional for the published port
- [ ] Set `TRANSCRIPTION_APP_PASSWORD` if sharing beyond a trusted LAN
- [ ] Do not commit `.env` or `.streamlit/secrets.toml`
