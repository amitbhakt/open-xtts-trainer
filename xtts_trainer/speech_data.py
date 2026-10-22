"""Download and prepare real human speech dataset for XTTS training.

Supports:
- Micro benchmark mode (8 samples, LJSpeech + JFK)
- Scaled training mode (5-10 minutes of continuous clean human speech: 100-200 clips)
"""

import os
import io
import argparse
import urllib.request
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

LONG_SPEECH_URL = "https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/4469669-10.mp3"

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
    output_dir: str = "sample_audio",
    num_samples: int = 8,
    sample_rate: int = 24000,
    max_duration_sec: float = 3.0,
    target_minutes: float = None,
) -> list[str]:
    """Download clean public-domain human speech samples, resample to 24kHz mono, and save as WAV.

    Args:
        output_dir: Directory to save processed .wav files.
        num_samples: Number of speech files to produce (e.g. 8 for quick test, 100 for 5 mins).
        sample_rate: Target sample rate (default 24,000 Hz for XTTS).
        max_duration_sec: Duration in seconds per training clip.
        target_minutes: If specified, overrides num_samples based on clip duration.

    Returns:
        List of generated WAV file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    if target_minutes is not None:
        num_samples = int((target_minutes * 60) / max_duration_sec)

    print(f"\n==================================================")
    print(f"Preparing {num_samples} Real Human Speech Clips (~{num_samples * max_duration_sec / 60:.1f} mins of audio)")
    print(f"Target format: {sample_rate}Hz 16-bit PCM mono WAV -> '{output_dir}'")
    print(f"==================================================")

    created_files = []

    # Scaled training mode (>8 samples or specified minutes)
    if num_samples > 8:
        print(f"Fetching 10-minute clean human speech audio (2.3 MB)...")
        try:
            req = urllib.request.Request(LONG_SPEECH_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()

            audio, orig_sr = sf.read(io.BytesIO(data))
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)

            tensor = torch.from_numpy(audio).float().unsqueeze(0)
            if orig_sr != sample_rate:
                tensor = F.resample(tensor, orig_sr, sample_rate)
            resampled = tensor.squeeze(0).numpy()

            chunk_size = int(max_duration_sec * sample_rate)
            max_available = len(resampled) // chunk_size
            total_to_slice = min(num_samples, max_available)

            for i in range(total_to_slice):
                out_filename = f"speech_{i+1:03d}.wav"
                out_path = os.path.join(output_dir, out_filename)
                clip = resampled[i * chunk_size : (i + 1) * chunk_size]

                # Peak normalize
                peak = np.max(np.abs(clip))
                if peak > 0:
                    clip = clip * (0.9 / peak)

                sf.write(out_path, clip, sample_rate, subtype="PCM_16")
                created_files.append(out_path)

            print(f"Successfully sliced {len(created_files)} clips ({len(created_files)*max_duration_sec/60:.2f} minutes) of human speech!")
            return created_files

        except Exception as e:
            print(f"Failed to fetch long audio ({e}). Falling back to multi-source samples.")

    # Micro mode (<= 8 samples)
    sources = SAMPLE_SPEECH_SOURCES[:min(num_samples, len(SAMPLE_SPEECH_SOURCES))]
    for idx, item in enumerate(sources):
        out_filename = f"speech_{idx+1:02d}_{item['name']}.wav"
        out_path = os.path.join(output_dir, out_filename)

        try:
            req = urllib.request.Request(item["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()

            audio, orig_sr = sf.read(io.BytesIO(data))
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)

            tensor = torch.from_numpy(audio).float().unsqueeze(0)
            if orig_sr != sample_rate:
                tensor = F.resample(tensor, orig_sr, sample_rate)

            max_samples = int(max_duration_sec * sample_rate)
            if tensor.shape[1] > max_samples:
                tensor = tensor[:, :max_samples]

            processed_audio = tensor.squeeze(0).numpy()
            peak = np.max(np.abs(processed_audio))
            if peak > 0:
                processed_audio = processed_audio * (0.9 / peak)

            sf.write(out_path, processed_audio, sample_rate, subtype="PCM_16")
            created_files.append(out_path)
            duration = len(processed_audio) / sample_rate
            print(f"  [{idx+1}/{len(sources)}] Saved {out_filename} ({duration:.2f}s) - {item['description']}")

        except Exception as e:
            print(f"  [Warning] Failed to fetch {item['url']}: {e}.")

    print(f"\nPrepared {len(created_files)} speech files in '{output_dir}'.")
    return created_files


def main():
    parser = argparse.ArgumentParser(description="Download and prepare human speech dataset for XTTS training")
    parser.add_argument("--output_dir", type=str, default="sample_audio", help="Target output folder")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of speech samples (e.g. 8 for quick, 100 for 5 min)")
    parser.add_argument("--minutes", type=float, default=None, help="Target duration in minutes (e.g. 5.0 or 10.0)")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Target sample rate (default: 24000)")
    parser.add_argument("--duration", type=float, default=3.0, help="Clip duration in seconds (default: 3.0)")
    args = parser.parse_args()

    download_and_prepare_speech(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        sample_rate=args.sample_rate,
        max_duration_sec=args.duration,
        target_minutes=args.minutes,
    )


if __name__ == "__main__":
    main()
