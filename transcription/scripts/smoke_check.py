#!/usr/bin/env python3
"""Release / ops smoke checks for the transcription app.

Usage:
  python scripts/smoke_check.py              # local packaging checks
  python scripts/smoke_check.py --http       # also hit Streamlit health URL
  python scripts/smoke_check.py --docker     # require docker CLI + compose file
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "app.py",
    "config.py",
    "requirements.txt",
    "requirements.lock",
    "Dockerfile",
    "docker-compose.yml",
    ".streamlit/config.toml",
    "LICENSE",
    "RELEASE_CHECKLIST.md",
)


def check_required_files() -> list[str]:
    missing = [rel for rel in REQUIRED_FILES if not (ROOT / rel).is_file()]
    return [f"missing file: {name}" for name in missing]


def check_dockerfile() -> list[str]:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    errors: list[str] = []
    for needle in ("ffmpeg", "streamlit", "HEALTHCHECK", "_stcore/health", "EXPOSE 8501"):
        if needle not in text:
            errors.append(f"Dockerfile missing expected content: {needle}")
    return errors


def check_compose() -> list[str]:
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    errors: list[str] = []
    for needle in ("whisper-cache", "./temp:/app/temp", "./output:/app/output", "8501"):
        if needle not in text:
            errors.append(f"docker-compose.yml missing expected content: {needle}")
    return errors


def check_http(url: str, timeout: float = 5.0) -> list[str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace").strip().lower()
            if resp.status != 200:
                return [f"health check HTTP {resp.status} from {url}"]
            if body and "ok" not in body and body != "ok":
                # Streamlit returns plain "ok"; accept any 200 with empty/ok body.
                return [f"unexpected health body from {url}: {body!r}"]
    except urllib.error.URLError as exc:
        return [f"health check failed for {url}: {exc}"]
    except TimeoutError:
        return [f"health check timed out for {url}"]
    return []


def check_docker_cli() -> list[str]:
    if shutil.which("docker") is None:
        return ["docker CLI not found on PATH"]
    try:
        subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return [f"docker daemon unavailable: {exc}"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcription app smoke checks")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Probe Streamlit health endpoint (default http://127.0.0.1:8501/_stcore/health)",
    )
    parser.add_argument(
        "--health-url",
        default="http://127.0.0.1:8501/_stcore/health",
        help="Health URL used with --http",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Require a working docker CLI (does not build images)",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    errors.extend(check_required_files())
    if (ROOT / "Dockerfile").is_file():
        errors.extend(check_dockerfile())
    if (ROOT / "docker-compose.yml").is_file():
        errors.extend(check_compose())
    if args.docker:
        errors.extend(check_docker_cli())
    if args.http:
        errors.extend(check_http(args.health_url))

    if errors:
        print("SMOKE FAIL")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("SMOKE OK")
    print(f"  root={ROOT}")
    print(f"  files={len(REQUIRED_FILES)} required present")
    if args.http:
        print(f"  health={args.health_url}")
    if args.docker:
        print("  docker=available")
    return 0


if __name__ == "__main__":
    sys.exit(main())
