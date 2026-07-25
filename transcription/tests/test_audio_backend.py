"""Tests for ffmpeg / audio backend configuration."""

from utils.audio_backend import configure_audio_backend, ffmpeg_available


def test_configure_audio_backend_idempotent():
    path1 = configure_audio_backend()
    path2 = configure_audio_backend()
    assert path1 == path2


def test_ffmpeg_available_after_configure():
    configure_audio_backend()
    # imageio-ffmpeg bundle should make this True even without system ffmpeg in PATH
    assert ffmpeg_available() is True
