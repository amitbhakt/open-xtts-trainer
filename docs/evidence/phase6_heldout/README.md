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
| LibriVox, *Fables de La Fontaine*, opening announcement | `fr` | The segments of the first 14.46 s | Given in the script; **must be verified by listening** |
| LibriVox, *Märchen der Gebrüder Grimm*, chapter 214, opening announcement | `de` | The segments of the first 13.75 s | Same |

`a0001` and `a0002` are excluded because the calibration voices' reference clips were built from them, so the
held-out text is new as well as the speakers. Licenses: each ARCTIC speaker's own `COPYING` is downloaded from
festvox by the run into `licenses/rms_COPYING` and `licenses/jmk_COPYING` and pinned like the audio; the LibriVox
items' public-domain license, as archive.org publishes it, is recorded in
[`licenses/librivox_archive_org_license.md`](licenses/librivox_archive_org_license.md).

**LibriVox windows, chosen before the run (2026-10-27).** Each reading opens with a fixed announcement, then the
reader's credit or the title, which the transcript doesn't cover. So each window ends in the middle of the pause
after "librivox point org", found as the longest run of 20 ms frames 40 dB below the peak:
- French: pause 14.12–14.80 s, window 14.46 s. A Whisper check of the earlier 16 s window heard "Enregistré par …"
  after the announcement, which would have counted as untranscribed speech.
- German: the first draft used chapter 114, whose noise floor is only 15 dB below its peak, with no pause at all, so
  its credit couldn't be cut off and it isn't a clean recording. Of 12 chapters screened (first 25 s each), chapter
  214 is clean (floor −64 dB), its announcement matches the transcript word for word in Whisper's reading, and it
  pauses 11.9–15.6 s before the title; window 13.75 s.

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

---

## 6. Results (run 2026-10-27 09:53 UTC)

Run on an NVIDIA A100-SXM4-40GB GPU in the locked environment (`torch==2.14.1+cu130`, `torchaudio==2.11.0+cu130`,
CUDA 13.0, `environment_issues: []`), with transcripts verified (`unverified_transcripts_allowed: false`) and every
download pinned (`source_pins.json`). Full per-clip output: [`heldout_alignment_validation.json`](heldout_alignment_validation.json); two identical reruns below.

**Provenance note.** The protocol, frozen thresholds and script were committed before the run (`562cf21`). Setting
`transcript_verified: True` for the two LibriVox sources, after listening, was done in the working tree and committed
with these results, not before the run. The script doesn't record the commit it ran from; the run's code is
`562cf21` plus that two-line change.

**Rerun from a committed tree (2026-10-27).** To close that gap, the validation was rerun from `826bd1e`, in which
the verification flags, the results above and commit recording are all committed, on the same GPU type and locked
environment. Output: [`heldout_alignment_validation_rerun.json`](heldout_alignment_validation_rerun.json). It
records `is_dirty: false`. The commit SHA it records (`2812f6b…`) is the runner's container commit: Modal does not
upload `.git`, so the runner commits the uploaded tree on top of a clone. That container tree matches
`826bd1e`'s file for file (identical blobs), except the four runtime volume mount points (`runs`, `checkpoints`,
`xtts_base`, `eval/baselines`), which become symlinks. None of them holds code the run executes.

Every per-clip score, rejection, verdict and summary number in the rerun is identical to the run above. The only
difference is stronger: the first run *recorded* the source pins, and the rerun *verified* all six downloads
against them. The rerun is therefore the clean-provenance result. The first run stands as a replicate, and §6.1–6.4
hold for both.

**Run 3: the full record (2026-10-27).** Runs 1 and 2 recorded the source pins, git state and torch versions, but
not the other inputs that earlier evidence (runs 4 and 6, ablation A3) records. Commit `f4106c5` adds them to the
report (format version 2), changing nothing in the analysis. Run 3 ran from that commit on the same GPU type.
Output: [`heldout_alignment_validation_run3.json`](heldout_alignment_validation_run3.json). It records:

| Record | Value | Check |
|---|---|---|
| `vocab.json` (the validator tokenizes every clip with it) | `92826087…` | Matches `artifacts.lock.json`; the script now refuses a mismatch |
| MMS_FA checkpoint (`ctc_alignment_mling_uroman/model.pt`, 1,262,047,414 bytes) | `20ef1296…` | No published digest exists. The container's fresh download equals an independent local download, byte for byte |
| Code fingerprint (`eval/harness.get_code_fingerprint`, 69 files) | `cb069f14…` | Equals the fingerprint of `f4106c5` computed locally |
| Container tree vs `f4106c5` | — | Identical blobs except the four volume mount points, as in the rerun |
| GPU, `transformers` | NVIDIA A100-SXM4-40GB, 5.20.2 | — |
| `is_dirty` | `false` | — |
| Source pins | all six `verified` | — |

Every per-clip score, rejection, verdict and summary number is identical to runs 1 and 2. Run 3 is the reference
record for this section; runs 1 and 2 are replicates.

### 6.1 Clean acceptance

| Language | Accepted | 95% Clopper–Pearson CI |
|---|---|---|
| English (`rms`, `jmk`) | 59/60 (98.3%) | [0.911, 1.000] |
| French | 4/4 | [0.398, 1.000] |
| German | 3/3 | [0.292, 1.000] |
| All | 66/67 (98.5%) | [0.920, 1.000] |

The one rejection is `arctic_jmk` `arctic_a0022` ("Hardly were our plans made public"): the word "our" scored 0.022,
below the 0.05 floor; the validator agrees. Both silence-trimming controls passed on all 17 mutated clips.

### 6.2 Mutations

Pre-registered criterion: a corruption counts as detected only where the rejection rate is at least 80%.

| Mutation | Rejected / total | Rejection rate (95% CI) | Expected | Verdict |
|---|---|---|---|---|
| `head_speech_cut_600ms` | 15/15 | 100.0% [0.782, 1.000] | reject | Detected |
| `head_speech_cut_300ms` | 15/16 | 93.8% [0.698, 0.998] | reject | Detected |
| `head_speech_cut_150ms` | 8/16 | 50.0% [0.247, 0.754] | reject | Not detected (limitation) |
| `tail_speech_cut_600ms` | 14/15 | 93.3% [0.681, 0.998] | reject | Detected |
| `tail_speech_cut_300ms` | 7/16 | 43.8% [0.198, 0.701] | reject | Not detected (limitation) |
| `tail_speech_cut_150ms` | 2/16 | 12.5% [0.016, 0.384] | reject | Not detected (limitation) |
| `tail_speech_cut_50ms` | 0/17 | 0.0% [0.000, 0.195] | reject | Not detected (limitation) |
| `insert_head_word` | 17/17 | 100.0% [0.805, 1.000] | reject | Detected |
| `insert_mid_word` | 13/17 | 76.5% [0.501, 0.932] | reject | Not detected (limitation; below 80%) |
| `insert_tail_word` | 11/17 | 64.7% [0.383, 0.858] | reject | Not detected (limitation) |
| `spillover_0.50s` | 12/15 | 80.0% [0.519, 0.957] | reject | Detected (at the bar) |
| `spillover_0.05s` | 1/15 | 6.7% | pass | 14/15 passed as expected |
| `control_trim_head_silence`, `control_trim_tail_silence` | 0/17 each | 0% | pass | All passed |

### 6.3 Pre-registered verdict

- **English: validated** for clean acceptance (≥ 90%, both controls pass on every clip). This is out-of-sample
  evidence (`[E1]`) for the false-rejection rate on clean English speech: at most about 9% at 95% confidence.
- **French and German: not validated.** 4 and 3 clean clips are too few, as section 4 states in advance; their
  results are reported only.
- **Detection: reliable only for large corruptions.** Head cuts of 300 ms or more, the 600 ms tail cut, an inserted
  opening word, and 0.5 s of spillover (exactly at the 80% bar, with a wide interval). Tail cuts up to 300 ms, the
  150 ms head cut, and inserted mid- or end-of-sentence words are limitations: they often pass.

### 6.4 Exploratory observations (not pre-registered; not for adopting new thresholds)

These come from the recorded per-word scores, after the results were seen. They suggest hypotheses for a new,
separately held-out run; adopting a rule tuned on this data would make it in-sample again.
- The 0.05 floor makes almost every rejection; the weak-word rule (at most 3 below 0.2) makes none. 65 of 67 clean
  clips have no weak word and 2 have one.
- Corruptions that pass usually have exactly one weak word, with lowest scores around 0.07–0.14.
- Unexplained speech reaches 0.081 s on clean clips, and the 0.5 s spillovers that pass sit around 0.08 s: the 0.1 s
  limit has no margin.
- A 150 ms tail cut leaves the last word scoring about 0.48 (median): the aligner fits the shortened word.
- On this data, a 0.10 floor would reject 13 of the 63 missed corruptions with no extra clean rejection, but would
  reject the in-sample accented speaker (`arctic_ksp`, lowest score 0.062). A boundary check (clips that end while
  speech is still sounding) is a candidate for the tail cuts; it needs word end times, which this run doesn't log.
