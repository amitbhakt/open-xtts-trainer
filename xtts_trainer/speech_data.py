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
        "text": "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition",
        "speaker": "ljspeech",
        "lang": "en",
    },
    {
        "name": "ljspeech_02",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0002.wav",
        "description": "LJSpeech (Linda Johnson) - Short phrase",
        "text": "in being comparatively modern.",
        "speaker": "ljspeech",
        "lang": "en",
    },
    {
        "name": "ljspeech_03",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0003.wav",
        "description": "LJSpeech (Linda Johnson) - Multi-clause sentence",
        "text": "For although the Chinese took impressions from wood blocks engraved in relief for centuries before the Woodcutters of the Netherlands, by a similar process",
        "speaker": "ljspeech",
        "lang": "en",
    },
    {
        "name": "ljspeech_04",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0004.wav",
        "description": "LJSpeech (Linda Johnson) - Narrative cadence",
        "text": "produced the block books, which were the immediate predecessors of the true printed book,",
        "speaker": "ljspeech",
        "lang": "en",
    },
    {
        "name": "ljspeech_05",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0005.wav",
        "description": "LJSpeech (Linda Johnson) - Historical text",
        "text": "the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing.",
        "speaker": "ljspeech",
        "lang": "en",
    },
    {
        "name": "ljspeech_06",
        "url": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0006.wav",
        "description": "LJSpeech (Linda Johnson) - Phonetic variety",
        "text": "And it is worth mention in passing that, as an example of fine typography,",
        "speaker": "ljspeech",
        "lang": "en",
    },
    {
        "name": "jfk_speech_01",
        "url": "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac",
        "description": "JFK Presidential Inaugural Address - First clause",
        "text": "And so, my fellow Americans: ask not what your country can do for you,",
        "speaker": "jfk",
        "lang": "en",
        "time_slice": (0.0, 5.5),
    },
    {
        "name": "jfk_speech_02",
        "url": "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac",
        "description": "JFK Presidential Inaugural Address - Second clause",
        "text": "ask what you can do for your country.",
        "speaker": "jfk",
        "lang": "en",
        "time_slice": (5.5, 11.0),
    },
]


def segment_audio_on_silence(
    audio: np.ndarray,
    sample_rate: int,
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 10.0,
    silence_thresh_db: float = -35.0,
) -> list[tuple[int, int]]:
    """Segment continuous speech audio at natural pauses to avoid slicing words mid-syllable."""
    frame_len = int(0.02 * sample_rate)  # 20ms
    hop_len = int(0.01 * sample_rate)    # 10ms
    num_frames = (len(audio) - frame_len) // hop_len + 1
    if num_frames <= 0:
        return [(0, len(audio))]

    shape = (num_frames, frame_len)
    strides = (audio.strides[0] * hop_len, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
    rms = np.sqrt(np.mean(frames ** 2, axis=-1) + 1e-9)
    peak_rms = np.max(rms)

    chunk_size = int(max_duration_sec * sample_rate)
    if peak_rms <= 1e-6:
        total_chunks = max(1, len(audio) // chunk_size)
        return [(i * chunk_size, min(len(audio), (i + 1) * chunk_size)) for i in range(total_chunks)]

    db = 20 * np.log10(rms / peak_rms)
    is_silent = db < silence_thresh_db

    # If there is virtually no silence (e.g. continuous synthetic test tone), chunk evenly
    if not np.any(is_silent):
        total_chunks = max(1, len(audio) // chunk_size)
        return [(i * chunk_size, min(len(audio), (i + 1) * chunk_size)) for i in range(total_chunks)]

    segments = []
    min_samples = int(min_duration_sec * sample_rate)
    max_samples = int(max_duration_sec * sample_rate)
    current_pos = 0

    while current_pos < len(audio):
        target_end = current_pos + max_samples
        if target_end >= len(audio):
            if len(audio) - current_pos >= int(1.0 * sample_rate):
                segments.append((current_pos, len(audio)))
            break

        # Search for silence in the window [current_pos + min_samples, target_end]
        search_start_sample = current_pos + min_samples
        search_start_frame = search_start_sample // hop_len
        search_end_frame = min(num_frames - 1, target_end // hop_len)

        split_sample = target_end
        if search_end_frame > search_start_frame:
            sub_silent = np.where(is_silent[search_start_frame:search_end_frame])[0]
            if len(sub_silent) > 0:
                split_frame = search_start_frame + sub_silent[len(sub_silent) // 2]
                split_sample = split_frame * hop_len

        segments.append((current_pos, split_sample))
        current_pos = split_sample

    return segments


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

    slices = segment_audio_on_silence(
        resampled,
        sample_rate=sample_rate,
        min_duration_sec=min(2.0, max_duration_sec),
        max_duration_sec=max_duration_sec,
    )
    created_files = []

    print(f"\n==================================================")
    print(f"Preparing speech from user recording: {input_file}")
    print(f"Total audio length: {len(resampled) / sample_rate:.2f}s -> {len(slices)} clips")
    print(f"Target format: {sample_rate}Hz 16-bit PCM mono WAV -> '{output_dir}'")
    print(f"==================================================")

    for i, (start_idx, end_idx) in enumerate(slices):
        out_filename = f"speech_{i+1:03d}.wav"
        out_path = os.path.join(output_dir, out_filename)
        clip = resampled[start_idx:end_idx]

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
        transcripts = [transcript] * len(created_files)
    else:
        # Transcribe clips using local Whisper ASR with correct sample rate
        print("Transcribing clips with local Whisper ASR to produce genuine ground-truth text...")
        try:
            from transformers import pipeline
            asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device="cpu")
            for path in created_files:
                clip_audio, sr = sf.read(path)
                res = asr({"raw": clip_audio, "sampling_rate": sr}, generate_kwargs={"language": "english", "task": "transcribe"})
                txt = res.get("text", "").strip()
                transcripts.append(txt if txt else f"Audio recording clip {len(transcripts)+1}.")
        except Exception as e:
            logger.warning(f"Whisper automated transcription unavailable ({e}). Using index prompt.")
            transcripts = [f"Speech clip {i+1}." for i in range(len(created_files))]

    with open(metadata_csv_path, "w", encoding="utf-8") as f:
        for i, path in enumerate(created_files):
            fname = os.path.basename(path)
            t = transcripts[i] if i < len(transcripts) else f"Training speech prompt {i+1}."
            f.write(f"{fname}|{t}|user_speaker|en\n")

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

            slices = segment_audio_on_silence(
                resampled,
                sample_rate=sample_rate,
                min_duration_sec=min(3.0, max_duration_sec),
                max_duration_sec=max(6.0, max_duration_sec),
            )
            total_to_slice = min(num_samples, len(slices))

            for i in range(total_to_slice):
                out_filename = f"speech_{i+1:03d}.wav"
                out_path = os.path.join(output_dir, out_filename)
                start_idx, end_idx = slices[i]
                clip = resampled[start_idx:end_idx]

                # Peak normalize
                peak = np.max(np.abs(clip))
                if peak > 0:
                    clip = clip * (0.9 / peak)

                sf.write(out_path, clip, sample_rate, subtype="PCM_16")
                created_files.append(out_path)

            print(f"Successfully sliced {len(created_files)} clips at natural pause boundaries of human speech!")
            print(f"Transcribing {len(created_files)} clips with local Whisper ASR for ground-truth alignment...")
            transcripts = []
            try:
                from transformers import pipeline
                asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device="cpu")
                for p in created_files:
                    clip_audio, sr = sf.read(p)
                    res = asr({"raw": clip_audio, "sampling_rate": sr}, generate_kwargs={"language": "english", "task": "transcribe"})
                    txt = res.get("text", "").strip()
                    if not txt:
                        raise RuntimeError(f"Whisper produced empty transcription for {p}")
                    transcripts.append(txt)
            except Exception as e:
                raise RuntimeError(
                    f"Automated Whisper speech transcription failed ({e}). "
                    f"Genuine paired transcripts are required for Stage 2 training."
                ) from e

            metadata_csv_path = os.path.join(output_dir, "metadata.csv")
            with open(metadata_csv_path, "w", encoding="utf-8") as f:
                for i, path in enumerate(created_files):
                    f.write(f"{os.path.basename(path)}|{transcripts[i]}|speaker_1|en\n")
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

            # Apply time slice if specified (e.g. multi-clause utterances)
            if "time_slice" in item:
                start_sec, end_sec = item["time_slice"]
                start_idx = int(start_sec * sample_rate)
                end_idx = min(tensor.shape[1], int(end_sec * sample_rate))
                tensor = tensor[:, start_idx:end_idx]

            # Keep complete utterance so text and audio match 1:1; only cap if exceeds XTTS maximum (~25s)
            max_samples = int(25.0 * sample_rate)
            if tensor.shape[1] > max_samples:
                tensor = tensor[:, :max_samples]

            processed_audio = tensor.squeeze(0).numpy()
            peak = np.max(np.abs(processed_audio))
            if peak > 0:
                processed_audio = processed_audio * (0.9 / peak)

            sf.write(out_path, processed_audio, sample_rate, subtype="PCM_16")
            created_files.append(out_path)
            metadata_entries.append((out_filename, item["text"], item.get("speaker", "default"), item.get("lang", "en")))
            duration = len(processed_audio) / sample_rate
            print(f"  [{idx+1}/{len(sources)}] Saved {out_filename} ({duration:.2f}s) - {item['description']}")

        except Exception as e:
            print(f"  [Warning] Failed to fetch {item['url']}: {e}.")

    # Write ground-truth metadata.csv
    metadata_csv_path = os.path.join(output_dir, "metadata.csv")
    with open(metadata_csv_path, "w", encoding="utf-8") as f:
        for fname, text, spk, lang in metadata_entries:
            f.write(f"{fname}|{text}|{spk}|{lang}\n")
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
