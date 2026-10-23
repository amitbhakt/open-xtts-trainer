"""Tests for speech_data preparation and 30-second rapid verification."""

import os
import shutil
import tempfile
import numpy as np
import soundfile as sf
import pytest

from xtts_trainer.speech_data import prepare_speech_from_file


@pytest.fixture
def temp_workspace():
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_prepare_speech_from_30s_file(temp_workspace):
    """Verify that a 30-second user recording is sliced into 10 clips of 3 seconds each with normalized format."""
    # 1. Create a synthetic 30-second recording at 44.1kHz stereo
    sr = 44100
    duration_sec = 30.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Stereo sine wave
    audio_ch1 = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio_ch2 = 0.5 * np.cos(2 * np.pi * 440 * t)
    stereo_audio = np.stack([audio_ch1, audio_ch2], axis=-1)

    input_file = os.path.join(temp_workspace, "my_30s_voice.wav")
    sf.write(input_file, stereo_audio, sr)

    output_dir = os.path.join(temp_workspace, "sample_audio")
    created_clips = prepare_speech_from_file(
        input_file=input_file,
        output_dir=output_dir,
        sample_rate=24000,
        max_duration_sec=3.0,
    )

    # 2. Check that exactly 10 clips were created
    assert len(created_clips) == 10
    assert os.path.exists(os.path.join(output_dir, "speech_001.wav"))
    assert os.path.exists(os.path.join(output_dir, "speech_010.wav"))

    # 3. Verify each clip properties
    for clip_path in created_clips:
        data, clip_sr = sf.read(clip_path)
        assert clip_sr == 24000
        assert data.ndim == 1  # Mono
        # Duration is 3.0 seconds = 72,000 samples
        assert len(data) == 72000
        assert np.max(np.abs(data)) <= 0.95
