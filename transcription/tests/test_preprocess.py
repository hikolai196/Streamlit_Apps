"""Tests for audio preprocessing."""

import numpy as np
import soundfile as sf

from modules.preprocess import preprocess_audio


def test_preprocess_converts_to_mono_16k_wav(tmp_path):
    input_path = tmp_path / "input.wav"
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    stereo = np.column_stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2)
    sf.write(str(input_path), stereo, sr)

    output_path = preprocess_audio(input_path)

    assert output_path.exists()
    assert output_path.suffix == ".wav"

    audio, out_sr = sf.read(str(output_path))
    assert out_sr == 16000
    assert audio.ndim == 1 or (audio.ndim == 2 and audio.shape[1] == 1)
