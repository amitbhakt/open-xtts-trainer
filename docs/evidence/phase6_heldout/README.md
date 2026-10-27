# Phase 6 held-out alignment validation: protocol and pre-registration

Out-of-sample validation of the alignment rejection thresholds (`QualityConfig`). The thresholds were calibrated
in-sample on the 8 recordings in `docs/evidence/phase6/` (`[E5]`); this run measures them, frozen, on speakers and
text that played no part in setting them. It reopens and is meant to close item 11 of the remediation plan.

**This file and `scripts/validate_alignment_heldout.py` are committed before the run.** That commit is the
pre-registration: what is measured, how it is labelled and how it will be read are fixed below, and nothing here
changes after the results are seen. Anything that does change is reported as a deviation.

---

## 1. Frozen thresholds

From the in-sample calibration, unchanged for this run:

| Setting | Value | Role |
|---|---|---|
| `min_word_score` | 0.05 | No re-aligned word may score below this |
| `weak_word_score` | 0.20 | A word below this counts as weak |
| `max_weak_words` | 3 | At most this many weak words |
| `max_unexplained_speech_s` | 0.10 s | Speech the transcript doesn't account for |
| `min_gap_s`, `edge_pad_s` (segmenting) | 0.15 s, 0.20 s | Cut only at pauses; keep 0.2 s of silence at each clip end |

## 2. Held-out sources

None of these speakers is in the calibration set (`jfk`, `arctic_slt`, `arctic_ksp`, `arctic_bdl`, `arctic_clb`,
`arctic_awb`, `ljspeech`, `spanish_cuentos_male`) or among the evaluation reference voices.

| Source | Language | Clean clips | Transcript |
|---|---|---|---|
| CMU ARCTIC `rms` (US English, male) | `en` | Utterances `arctic_a0003`–`arctic_a0022` (20) | The corpus's own prompt file (`etc/txt.done.data`) |
| CMU ARCTIC `jmk` (Canadian English, male) | `en` | Utterances `arctic_a0003`–`arctic_a0022` (20) | Same |
| LibriVox, *Fables de La Fontaine*, opening announcement | `fr` | The segments of the first 16 s | Given in the script; **must be verified by listening** |
| LibriVox, *Märchen der Gebrüder Grimm*, opening announcement | `de` | The segments of the first 17 s | Same |

`a0001` and `a0002` are excluded because the calibration voices' reference clips were built from them, so the
held-out text is new as well as the speakers. Licenses: each ARCTIC speaker's own `COPYING` is downloaded from
festvox by the run into `licenses/rms_COPYING` and `licenses/jmk_COPYING` and pinned like the audio; the LibriVox
items' public-domain license, as archive.org publishes it, is recorded in
[`licenses/librivox_archive_org_license.md`](licenses/librivox_archive_org_license.md).

**Download pins.** Every archive's SHA-256 is recorded in `source_pins.json` on the first run and verified on every
later one; a mismatch stops the run.

## 3. Measurements

For every clean clip:
- **Accepted** when preparation keeps the segment *and* re-alignment of the clip finds no problem (the validator's
  check, `validate.realignment_checker`). A clip preparation drops counts as rejected.
- The validator's full result on the kept clip (`validator_ok`), as training would run it.
- Per-word scores and unexplained speech.

For the first 5 clean clips of each source, a mutation suite. Each clip keeps 0.2 s of silence at both ends, so every
label is defined by speech removed or added, **measured from the aligned words, not the clip edges**:

| Mutation | What changes | Expected |
|---|---|---|
| `control_trim_tail_silence`, `control_trim_head_silence` | Silence trimmed to 50 ms beyond the words; no speech lost | pass |
| `tail_speech_cut_{50,150,300,600}ms` | That much speech removed from the end | reject |
| `head_speech_cut_{150,300,600}ms` | That much speech removed from the start | reject |
| `insert_head_word`, `insert_mid_word`, `insert_tail_word` | A word added to the transcript that the audio doesn't contain | reject |
| `spillover_0.05s` | 50 ms of the next utterance's speech appended | pass (below the 0.10 s limit) |
| `spillover_0.50s` | 500 ms of the next utterance's speech appended | reject |

Cuts that would leave under 0.5 s of speech are skipped and not counted. Every mutated clip's per-word scores and
unexplained speech are recorded.

## 4. Pre-registered analysis

Reported exactly as below, whatever the results:
1. **Clean acceptance**: accepted / total, overall and per language, with a 95% Clopper–Pearson interval. The
   in-sample figure was 8/8; a lower held-out figure is reported as the out-of-sample false-positive rate.
2. **Detection by mutation**: rejected / total for each mutation, with intervals. For speech cuts, rates are shown
   as a function of speech removed (a detection curve), not pooled into one "correct" count.
3. **Controls**: both must pass on every clip. A control that fails means the thresholds reject correct clips that
   differ only in silence, and is reported as a false positive.
4. **Score margins**: the distribution of the lowest word score and of the weak-word count on clean clips versus
   each mutation, from the recorded per-word scores.

Interpretation fixed in advance: the thresholds count as **validated for a language** only if clean acceptance is
at least 90% there and both controls pass on every clip. They count as **detecting a corruption** only at the
severities where the rejection rate is at least 80%. Anything else is reported as a limitation, not adjusted.
There are 40 English clean clips and only a few French and German ones, so the French and German results are
reported, but too few to validate those languages on their own.

## 5. Running it

```bash
uv sync --frozen --extra dev                                      # the locked environment (torch 2.14.1)
python scripts/fetch_base_artifacts.py --dest xtts_base --only vocab.json config.json mel_stats.pth
python scripts/validate_alignment_heldout.py --check_transcripts  # compare given vs Whisper for fr/de
# Listen to each fr/de window, correct its transcript if needed, set transcript_verified: True, commit.
python scripts/validate_alignment_heldout.py --device cuda         # writes heldout_alignment_validation.json
```

The script refuses to run outside the versions pinned in `uv.lock` (`--allow_unlocked_env` exists for development
and is recorded in the output), and refuses unverified transcripts (`--allow_unverified_transcripts`, also
recorded). Output: `heldout_alignment_validation.json` in this directory, plus `source_pins.json` on the first run.
