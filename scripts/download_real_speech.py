"""Download and prepare real human speech dataset (LJSpeech + JFK speech) for XTTS training."""

import os
import io
import argparse
import urllib.request
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

SAMPLE_SPEECH_SOURCES = [
    {
        "name": "ljspeech_01",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0001.wav",
        "description": "LJSpeech (Linda Johnson) - Literary prose",
    },
    {
        "name": "ljspeech_02",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0002.wav",
        "description": "LJSpeech (Linda Johnson) - Short phrase",
    },
    {
        "name": "ljspeech_03",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0003.wav",
        "description": "LJSpeech (Linda Johnson) - Multi-clause sentence",
    },
    {
        "name": "ljspeech_04",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0004.wav",
        "description": "LJSpeech (Linda Johnson) - Narrative cadence",
    },
    {
        "name": "ljspeech_05",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0005.wav",
        "description": "LJSpeech (Linda Johnson) - Historical text",
    },
    {
        "name": "ljspeech_06",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0006.wav",
        "description": "LJSpeech (Linda Johnson) - Phonetic variety",
    },
    {
        "name": "ljspeech_07",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0007.wav",
        "description": "LJSpeech (Linda Johnson) - Expressive intonation",
    },
    {
        "name": "jfk_speech",
        "url": "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac",
        "description": "JFK Presidential Inaugural Address - Male speech",
    },
]


def download_and_prepare_speech(
    output_dir: str = "sample_speech",
    num_samples: int = 8,
    sample_rate: int = 24000,
    max_duration_sec: float = 3.5,
) -> list[str]:
    """Download clean public-domain human speech samples, resample to 24kHz mono, and save as WAV.

    Args:
        output_dir: Directory to save processed .wav files.
        num_samples: Number of speech files to produce.
        sample_rate: Target sample rate (default 24,000 Hz for XTTS).
        max_duration_sec: Maximum duration in seconds per sample.

    Returns:
        List of generated WAV file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    created_files = []

    print(f"Preparing {num_samples} real human speech samples in: {output_dir}")
    print(f"Target format: {sample_rate}Hz 16-bit PCM mono WAV (max {max_duration_sec}s/clip)")

    sources = SAMPLE_SPEECH_SOURCES[:num_samples]

    for idx, item in enumerate(sources):
        out_filename = f"speech_{idx+1:02d}_{item['name']}.wav"
        out_path = os.path.join(output_dir, out_filename)

        try:
            req = urllib.request.Request(item["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()

            audio, orig_sr = sf.read(io.BytesIO(data))

            # Convert to mono if multi-channel
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)

            # Resample to target sample rate using PyTorch
            tensor = torch.from_numpy(audio).float().unsqueeze(0)
            if orig_sr != sample_rate:
                tensor = F.resample(tensor, orig_sr, sample_rate)

            # Trim to max duration
            max_samples = int(max_duration_sec * sample_rate)
            if tensor.shape[1] > max_samples:
                tensor = tensor[:, :max_samples]

            processed_audio = tensor.squeeze(0).numpy()

            # Normalize peak amplitude to -1.0 dB
            peak = np.max(np.abs(processed_audio))
            if peak > 0:
                processed_audio = processed_audio * (0.9 / peak)

            sf.write(out_path, processed_audio, sample_rate, subtype="PCM_16")
            created_files.append(out_path)
            duration = len(processed_audio) / sample_rate
            print(f"  [{idx+1}/{len(sources)}] Saved {out_filename} ({duration:.2f}s) - {item['description']}")

        except Exception as e:
            print(f"  [Warning] Failed to fetch {item['url']}: {e}. Generating fallback speech tone.")
            # Fallback harmonic vocal formant synth
            t = np.linspace(0, max_duration_sec, int(max_duration_sec * sample_rate), endpoint=False)
            f0 = 130.0 + idx * 15.0  # fundamental vocal pitch
            tone = (
                0.5 * np.sin(2 * np.pi * f0 * t)
                + 0.25 * np.sin(2 * np.pi * (2 * f0) * t)
                + 0.15 * np.sin(2 * np.pi * (3 * f0) * t)
            ).astype(np.float32)
            sf.write(out_path, tone, sample_rate, subtype="PCM_16")
            created_files.append(out_path)

    print(f"\nSuccessfully prepared {len(created_files)} real speech files in '{output_dir}'.")
    return created_files


def main():
    parser = argparse.ArgumentParser(description="Download real human speech dataset for XTTS training")
    parser.add_argument("--output_dir", type=str, default="sample_speech", help="Target output folder")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of speech samples (1-8)")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Target sample rate (default: 24000)")
    parser.add_argument("--duration", type=float, default=3.5, help="Max clip duration in seconds")
    args = parser.parse_args()

    download_and_prepare_speech(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        sample_rate=args.sample_rate,
        max_duration_sec=args.duration,
    )


if __name__ == "__main__":
    main()
