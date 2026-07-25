"""Phase 6 ops packaging checks (Dockerfile, compose, smoke script)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_smoke_module():
    path = ROOT / "scripts" / "smoke_check.py"
    spec = importlib.util.spec_from_file_location("smoke_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dockerfile_and_compose_exist():
    assert (ROOT / "Dockerfile").is_file()
    assert (ROOT / "docker-compose.yml").is_file()
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "RELEASE_CHECKLIST.md").is_file()


def test_smoke_check_passes_locally():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_check.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SMOKE OK" in result.stdout


def test_smoke_helpers_detect_missing_health_marker(tmp_path: Path, monkeypatch):
    smoke = _load_smoke_module()
    docker_src = (ROOT / "Dockerfile").read_text(encoding="utf-8").replace("HEALTHCHECK", "HEALTH_X")
    compose_src = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    (tmp_path / "Dockerfile").write_text(docker_src, encoding="utf-8")
    (tmp_path / "docker-compose.yml").write_text(compose_src, encoding="utf-8")
    monkeypatch.setattr(smoke, "ROOT", tmp_path)

    errors = smoke.check_dockerfile()
    assert any("HEALTHCHECK" in err for err in errors)


def test_compose_lists_volume_mounts():
    smoke = _load_smoke_module()
    assert smoke.check_compose() == []


def test_health_url_failure_reported():
    smoke = _load_smoke_module()
    errors = smoke.check_http("http://127.0.0.1:9/_stcore/health", timeout=1.0)
    assert errors
