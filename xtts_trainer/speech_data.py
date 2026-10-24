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

LONG_SPEECH_URL_10M = "https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/4469669-10.mp3"
LONG_SPEECH_URL_2H = "https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/4469669.mp3"

SAMPLE_SPEECH_SOURCES = [
    {
        "name": "ljspeech_01",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0001.wav",
        "description": "LJSpeech (Linda Johnson) - Literary prose",
        "text": "Printing, in the only sense with which we are at present concerned, differs from most if not from all the other arts and crafts represented in the Exhibition",
        "lang": "en",
    },
    {
        "name": "ljspeech_02",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0002.wav",
        "description": "LJSpeech (Linda Johnson) - Short phrase",
        "text": "in being comparatively modern.",
        "lang": "en",
    },
    {
        "name": "ljspeech_03",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0003.wav",
        "description": "LJSpeech (Linda Johnson) - Multi-clause sentence",
        "text": "For although the Chinese took impressions from wood blocks engraved in relief for centuries before the Woodcutters of the Netherlands, by a similar process",
        "lang": "en",
    },
    {
        "name": "ljspeech_04",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0004.wav",
        "description": "LJSpeech (Linda Johnson) - Narrative cadence",
        "text": "produced the block books, which were the immediate predecessors of the true printed book,",
        "lang": "en",
    },
    {
        "name": "ljspeech_05",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0005.wav",
        "description": "LJSpeech (Linda Johnson) - Historical text",
        "text": "the invention of moveable sorts in the middle of the fifteenth century may justly be considered as the invention of the art of printing.",
        "lang": "en",
    },
    {
        "name": "ljspeech_06",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0006.wav",
        "description": "LJSpeech (Linda Johnson) - Phonetic variety",
        "text": "And it is worth mention in passing that, as an example of fine typography,",
        "lang": "en",
    },
    {
        "name": "ljspeech_07",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0007.wav",
        "description": "LJSpeech (Linda Johnson) - Expressive intonation",
        "text": "the earliest book printed with movable types, the Gutenberg, or forty-two line Bible of about 1455,",
        "lang": "en",
    },
    {
        "name": "jfk_speech",
        "url": "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac",
        "description": "JFK Presidential Inaugural Address - Male speech",
        "text": "And so, my fellow Americans: ask not what your country can do for you — ask what you can do for your country.",
        "lang": "en",
    },
]


def prepare_speech_from_file(
    input_file: str,
    output_dir: str = "sample_audio",
    sample_rate: int = 24000,
    max_duration_sec: float = 3.0,
    transcript: str = None,
    transcript_file: str = None,
) -> list[str]:
    """Slice a user-provided recording into normalized 24kHz mono training clips.

    Args:
        input_file: Path to local audio file (e.g. 30-second recording).
        output_dir: Directory to save processed .wav clips and metadata.csv.
        sample_rate: Target sample rate (default 24000 Hz).
        max_duration_sec: Duration in seconds per training clip.
        transcript: Optional single transcript applied to all clips.
        transcript_file: Optional file containing one transcript per line.

    Returns:
        List of generated WAV file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input audio file not found: {input_file}")

    audio, orig_sr = sf.read(input_file)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    tensor = torch.from_numpy(audio).float().unsqueeze(0)
    if orig_sr != sample_rate:
        tensor = F.resample(tensor, orig_sr, sample_rate)
    resampled = tensor.squeeze(0).numpy()

    # Trim leading silence so speech_001.wav starts with active speech
    non_silent = np.where(np.abs(resampled) > 0.01)[0]
    if len(non_silent) > 0:
        lead_offset = max(0, non_silent[0] - int(0.05 * sample_rate))
        resampled = resampled[lead_offset:]

    chunk_size = int(max_duration_sec * sample_rate)
    total_clips = max(1, len(resampled) // chunk_size)
    created_files = []

    print(f"\n==================================================")
    print(f"Preparing speech from user recording: {input_file}")
    print(f"Total audio length: {len(resampled) / sample_rate:.2f}s -> {total_clips} clips ({max_duration_sec}s each)")
    print(f"Target format: {sample_rate}Hz 16-bit PCM mono WAV -> '{output_dir}'")
    print(f"==================================================")

    for i in range(total_clips):
        out_filename = f"speech_{i+1:03d}.wav"
        out_path = os.path.join(output_dir, out_filename)
        clip = resampled[i * chunk_size : (i + 1) * chunk_size]
        if len(clip) < chunk_size:
            clip = np.pad(clip, (0, chunk_size - len(clip)))

        peak = np.max(np.abs(clip))
        if peak > 0:
            clip = clip * (0.9 / peak)

        sf.write(out_path, clip, sample_rate, subtype="PCM_16")
        created_files.append(out_path)

    # Write metadata.csv pairing each clip with transcript
    metadata_csv_path = os.path.join(output_dir, "metadata.csv")
    transcripts = []
    if transcript_file and os.path.exists(transcript_file):
        with open(transcript_file, "r", encoding="utf-8") as tf:
            transcripts = [line.strip() for line in tf if line.strip()]
    elif transcript:
        transcripts = [transcript] * total_clips

    with open(metadata_csv_path, "w", encoding="utf-8") as f:
        for i, path in enumerate(created_files):
            fname = os.path.basename(path)
            t = transcripts[i] if i < len(transcripts) else f"Training speech prompt {i+1}."
            f.write(f"{fname}|{t}|en\n")

    print(f"Successfully created {len(created_files)} clips and wrote metadata to '{metadata_csv_path}'.")
    return created_files


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
        speech_url = LONG_SPEECH_URL_2H if ((target_minutes and target_minutes > 10) or num_samples > 200) else LONG_SPEECH_URL_10M
        url_label = "2-hour corpus (28 MB)" if speech_url == LONG_SPEECH_URL_2H else "10-minute corpus (2.3 MB)"
        print(f"Fetching clean human speech audio ({url_label})...")
        try:
            req = urllib.request.Request(speech_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()

            audio, orig_sr = sf.read(io.BytesIO(data))
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)

            tensor = torch.from_numpy(audio).float().unsqueeze(0)
            if orig_sr != sample_rate:
                tensor = F.resample(tensor, orig_sr, sample_rate)
            resampled = tensor.squeeze(0).numpy()

            # Trim leading silence so clips begin immediately with active speech
            non_silent = np.where(np.abs(resampled) > 0.01)[0]
            if len(non_silent) > 0:
                lead_offset = max(0, non_silent[0] - int(0.05 * sample_rate))
                resampled = resampled[lead_offset:]

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
            metadata_csv_path = os.path.join(output_dir, "metadata.csv")
            with open(metadata_csv_path, "w", encoding="utf-8") as f:
                for i, path in enumerate(created_files):
                    f.write(f"{os.path.basename(path)}|Human speech training clip number {i+1}.|en\n")
            print(f"Wrote metadata index to: {metadata_csv_path}")
            return created_files

        except Exception as e:
            print(f"Failed to fetch long audio ({e}). Falling back to multi-source samples.")

    # Micro mode (<= 8 samples)
    sources = SAMPLE_SPEECH_SOURCES[:min(num_samples, len(SAMPLE_SPEECH_SOURCES))]
    metadata_entries = []
    for idx, item in enumerate(sources):
        out_filename = f"speech_{idx+1:03d}.wav"
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
            metadata_entries.append((out_filename, item.get("text", "Sample human speech."), item.get("lang", "en")))
            duration = len(processed_audio) / sample_rate
            print(f"  [{idx+1}/{len(sources)}] Saved {out_filename} ({duration:.2f}s) - {item['description']}")

        except Exception as e:
            print(f"  [Warning] Failed to fetch {item['url']}: {e}.")

    # Write ground-truth metadata.csv
    metadata_csv_path = os.path.join(output_dir, "metadata.csv")
    with open(metadata_csv_path, "w", encoding="utf-8") as f:
        for fname, text, lang in metadata_entries:
            f.write(f"{fname}|{text}|{lang}\n")
    print(f"Wrote ground-truth paired transcripts to: {metadata_csv_path}")

    print(f"\nPrepared {len(created_files)} speech files in '{output_dir}'.")
    return created_files


def main():
    parser = argparse.ArgumentParser(description="Download and prepare human speech dataset for XTTS training")
    parser.add_argument("--input_file", "-i", type=str, default=None, help="Path to local audio recording file to slice and prepare")
    parser.add_argument("--output_dir", type=str, default="sample_audio", help="Target output folder")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of speech samples (e.g. 8 for quick, 100 for 5 min)")
    parser.add_argument("--minutes", type=float, default=None, help="Target duration in minutes (e.g. 0.5 for 30s test, 5.0 for 5 min)")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Target sample rate (default: 24000)")
    parser.add_argument("--duration", type=float, default=3.0, help="Clip duration in seconds (default: 3.0)")
    args = parser.parse_args()

    if args.input_file:
        prepare_speech_from_file(
            input_file=args.input_file,
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
            max_duration_sec=args.duration,
        )
    else:
        download_and_prepare_speech(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            sample_rate=args.sample_rate,
            max_duration_sec=args.duration,
            target_minutes=args.minutes,
        )


if __name__ == "__main__":
    main()
