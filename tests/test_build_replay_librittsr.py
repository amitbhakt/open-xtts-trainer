"""scripts/build_replay_librittsr.py: M1's replay set, on small fake LibriTTS-R archives (no network)."""

import io
import json
import os
import sys
import tarfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pytest
import soundfile as sf

from dataset_helpers import write_test_vocab
from scripts import build_replay_librittsr as replay

EVAL_SENTENCE = "The birch canoe slid on the smooth planks."  # en_01 of the frozen eval set
SPEAKERS_TSV = ("READER\tGENDER\tSUBSET\tNAME\n"
                "209\tF\tdev-clean\tTarget Reader\n"  # excluded even if it were in the subset
                "1\tF\tdev-clean\tA\n2\tM\tdev-clean\tB\n3\tF\tdev-clean\tC\n4\tM\tdev-clean\tD\n"
                "6\tF\tdev-clean\tF\n"
                "5\tF\ttrain-clean-100\tE\n")
TEXTS = ["the quiet river runs past the old mill", "a green field lies under the morning sky",
         "she walked slowly along the stone wall", "the wind moved through the tall grass",
         "he opened the wooden door and looked out", "a small boat drifted near the shore"]


def _add(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _wav_bytes(seconds, seed):
    buf = io.BytesIO()
    rng = np.random.RandomState(seed)
    sf.write(buf, (0.1 * rng.randn(int(24000 * seconds))).astype("float32"), 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _archives(root):
    doc = root / "doc.tar.gz"
    with tarfile.open(doc, "w:gz") as tar:
        _add(tar, "LibriTTS_R/speakers.tsv", SPEAKERS_TSV.encode())
    dev = root / "dev_clean.tar.gz"
    with tarfile.open(dev, "w:gz") as tar:
        _add(tar, "../evil.wav", b"outside")  # never extracted: it doesn't match the corpus layout
        for spk in ("1", "2", "3", "4", "6", "209"):
            for k in range(6):
                utt = f"{spk}_100_000{k:03d}_000000"
                # One too short and one eval-set sentence per speaker; four usable. Reader 6 has only short clips
                seconds = 1.0 if k == 0 or spk == "6" else 4.0
                text = EVAL_SENTENCE if k == 1 else TEXTS[k]
                _add(tar, f"LibriTTS_R/dev-clean/{spk}/100/{utt}.wav", _wav_bytes(seconds, int(spk) * 10 + k))
                _add(tar, f"LibriTTS_R/dev-clean/{spk}/100/{utt}.normalized.txt", text.encode())
    return doc, dev


@pytest.fixture
def fake_corpus(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    doc, dev = _archives(src)
    archives = {}
    for path in (doc, dev):
        md5, sha = replay.file_digests(str(path))
        archives[path.name] = {"md5": md5, "sha256": sha}
    monkeypatch.setattr(replay, "ARCHIVES", archives)
    monkeypatch.setattr(replay, "N_SPEAKERS", 3)
    monkeypatch.setattr(replay, "CLIPS_PER_SPEAKER", 2)
    vocab = write_test_vocab(str(tmp_path / "vocab.json"))
    return src.as_uri() + "/", vocab, tmp_path


def test_selection_follows_the_fixed_rule(fake_corpus):
    url_base, vocab, tmp = fake_corpus
    sel = replay.build(str(tmp / "out"), str(tmp / "cache"), vocab, url_base=url_base)
    readers = [s["reader"] for s in sel["speakers"]]
    assert "209" not in readers and len(readers) == 3
    assert [s["gender"] for s in sel["speakers"]] == ["F", "M", "F"]  # alternating, female first
    assert len(sel["clips"]) == 3 * 2
    assert all(c["text"] != EVAL_SENTENCE and replay.MIN_S <= c["duration_s"] <= replay.MAX_S for c in sel["clips"])
    reasons = " ".join(r["reason"] for r in sel["rejected"])
    assert "duration" in reasons  # the 1 s clips
    # Reader 6 is first in the female queue (seeded) but has no usable clip: skipped, recorded, never chosen
    assert replay.reader_queues(replay.read_speakers(str(tmp / "cache" / "doc.tar.gz")))["F"][0] == "6"
    assert "6" not in readers
    assert [s["reader"] for s in sel["skipped_readers"]] == ["6"] and "usable" in sel["skipped_readers"][0]["reason"]
    with open(tmp / "out" / "dataset.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    records = [r for r in records if "audio" in r]  # skip a header line, if any
    assert len(records) == 6 and {r["speaker"] for r in records} == {f"librittsr_{r}" for r in readers}
    assert len(os.listdir(tmp / "out" / "clips")) == 6  # rejected candidates' audio removed
    assert not os.path.exists(tmp / "cache" / "evil.wav") and not os.path.exists(tmp / "evil.wav")
    # Deterministic: the same archives and rule give the same selection
    again = replay.build(str(tmp / "out2"), str(tmp / "cache"), vocab, url_base=url_base)
    assert again["clips"] == sel["clips"] and again["speakers"] == sel["speakers"]


def test_eval_sentences_are_rejected_when_a_speaker_reaches_them(fake_corpus, monkeypatch):
    url_base, vocab, tmp = fake_corpus
    monkeypatch.setattr(replay, "CLIPS_PER_SPEAKER", 4)  # needs every usable clip, so the eval sentence is reached
    sel = replay.build(str(tmp / "out"), str(tmp / "cache"), vocab, url_base=url_base)
    assert all(c["text"] != EVAL_SENTENCE for c in sel["clips"])
    assert any("eval" in r["reason"] for r in sel["rejected"])


def test_archives_must_match_published_md5_and_pin(fake_corpus, monkeypatch):
    url_base, vocab, tmp = fake_corpus
    bad_md5 = {k: dict(v) for k, v in replay.ARCHIVES.items()}
    bad_md5["doc.tar.gz"]["md5"] = "0" * 32
    monkeypatch.setattr(replay, "ARCHIVES", bad_md5)
    with pytest.raises(RuntimeError, match="published"):
        replay.fetch("doc.tar.gz", str(tmp / "c1"), url_base)
    bad_sha = {k: dict(v) for k, v in replay.ARCHIVES.items()}
    bad_sha["doc.tar.gz"]["md5"] = replay.file_digests(str(tmp / "c1" / "doc.tar.gz"))[0]
    bad_sha["doc.tar.gz"]["sha256"] = "0" * 64
    monkeypatch.setattr(replay, "ARCHIVES", bad_sha)
    with pytest.raises(RuntimeError, match="pinned"):
        replay.fetch("doc.tar.gz", str(tmp / "c1"), url_base)


def test_reader_queues_exclude_and_are_seeded():
    speakers = {"209": "F", "1": "F", "3": "F", "2": "M"}
    queues = replay.reader_queues(speakers)
    assert sorted(queues["F"]) == ["1", "3"] and queues["M"] == ["2"]
    assert replay.reader_queues(speakers) == queues


def _raw_reader(raw, spk, texts):
    """A reader's extracted utterances (4 s each) with the given texts; returns the utterance ids."""
    raw.mkdir(parents=True, exist_ok=True)
    utts = []
    for k, text in enumerate(texts):
        utt = f"{spk}_200_{k:06d}_000000"
        (raw / f"{utt}.wav").write_bytes(_wav_bytes(4.0, 1000 + k))
        (raw / f"{utt}.normalized.txt").write_text(text)
        utts.append(utt)
    return utts


def test_a_reader_is_skipped_only_when_it_truly_lacks_usable_clips(tmp_path, monkeypatch):
    """Usable clips beyond the first batch still count: scanning goes on until enough pass or none are left."""
    monkeypatch.setattr(replay, "CLIPS_PER_SPEAKER", 2)
    vocab = write_test_vocab(str(tmp_path / "vocab.json"))
    texts = [EVAL_SENTENCE] * 14 + [TEXTS[0], TEXTS[2]]  # 14 unusable (eval sentence), 2 usable
    utts = _raw_reader(tmp_path / "raw", "7", texts)
    usable = {u for u, t in zip(utts, texts) if t != EVAL_SENTENCE}
    order = replay.candidate_order(utts, "7")
    batch = 3 * replay.CLIPS_PER_SPEAKER
    assert any(order.index(u) >= batch for u in usable)  # the case a single capped batch would wrongly skip
    (tmp_path / "out" / "clips").mkdir(parents=True)
    chosen, rejected = replay.usable_clips("7", utts, str(tmp_path / "raw"), str(tmp_path / "out"), vocab)
    assert {r.source.split()[-1] for r in chosen} == usable
    assert len(rejected) == 14 and all("eval" in r["reason"] for r in rejected)
    assert sorted(os.listdir(tmp_path / "out" / "clips")) == sorted(f"librittsr_{u}.wav" for u in usable)


def test_a_skipped_readers_usable_clips_are_recorded_and_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "CLIPS_PER_SPEAKER", 2)
    vocab = write_test_vocab(str(tmp_path / "vocab.json"))
    utts = _raw_reader(tmp_path / "raw", "8", [TEXTS[0], EVAL_SENTENCE])  # only one usable clip
    (tmp_path / "out" / "clips").mkdir(parents=True)
    chosen, rejected = replay.usable_clips("8", utts, str(tmp_path / "raw"), str(tmp_path / "out"), vocab)
    assert chosen == []
    reasons = {r["utterance"]: r["reason"] for r in rejected}
    assert reasons.keys() == set(utts) and "reader was skipped" in reasons[utts[0]]
    assert os.listdir(tmp_path / "out" / "clips") == []
