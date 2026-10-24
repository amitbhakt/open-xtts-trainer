# Colab run 2: Phase 0 + Phase 1 pilot, stratified (2026-10-24, Tesla T4)

Produced by `notebooks/phase0_phase1_colab.ipynb` (run-2 version, before Phase 6 was added) from an uploaded
working-tree zip, so `commit_sha` is `unknown`. Code fingerprint at setup: `8cbeb093…` over 48 files, the
same as the local tree when the zip was built. The Phase 6 alignment step (7b) was not part of this run.

## What it proves

| File | Proves |
|---|---|
| `artifact_verification.json` | All 6 upstream artifacts match `artifacts.lock.json` again, and the three-way `mel_stats` check passes. Identical to run 1 apart from the timestamp. |
| `environment_colab.json` | Same stack as run 1: Python 3.13.16, torch 2.11.0+cu130, transformers 5.20.2, `coqui-tts` at `03339e5b`. |
| `reference_voices.lock.json` | Reference voices rebuilt with `--update_lock`. Each lock entry records the transcript, the utterances used (a0001 + a0002, 6.6–7.5 s), and the SHA-256 of that speaker's CMU ARCTIC `COPYING` file. Copied to `eval/dataset/` as the lock every later run verifies against. The licenses differ only in copyright year (2003 vs 2005). |
| `determinism_check.json` | The smoke subset rerun matches exactly: difference 0.0 on every metric, and identical ASR text and duration for every utterance. |
| `pilot_en.json` | Stock XTTS-v2 baseline: English, 12 sentences (2 per category) × 5 voices × 2 seeds = 120 utterances. About 2.3 s synthesis per utterance. |

## Pilot results by category

| Category | WER | CER | SECS | UTMOS | Duration ratio | Too long |
|---|---|---|---|---|---|---|
| harvard | 0.000 | 0.000 | 0.559 | 4.05 | 0.98 | 0 |
| sibilant | 0.259 | 0.105 | 0.481 | 4.13 | 1.14 | 0 |
| numbers | 0.083 | 0.010 | 0.614 | 4.26 | 1.61 | 0 |
| long_complex | 0.000 | 0.000 | 0.615 | 4.06 | 0.94 | 0 |
| prosody_questions | 0.000 | 0.000 | 0.589 | 4.11 | 0.84 | 0 |
| short | 0.000 | 0.000 | 0.332 | 3.52 | 1.54 | 1 |

Overall: WER 0.057, CER 0.019, SECS 0.532 ± 0.024, UTMOS 4.02. Per voice, SECS ranges from 0.44 (slt) to 0.61 (awb).

## What the numbers do and don't say

- **Run 1's WER of 0.005 was an artifact of an easy subset.** Stratified sampling exposes errors: all the sibilant WER comes from the tongue twister en_12 ("The six sick sheiks' sixth sheep is sick"), which ASR also struggles with.
- **Some duration and similarity metrics are measurement problems, not model failures:**
  - The numbers category's duration ratio (1.61) counts the written digits, not the spoken words. "4,800,000" is 9 characters but takes about 2 s to say, so every en_22 utterance scores 1.8–2.1 while the ASR transcript is exact.
  - The single "too long" flag is "Yes." at 1.15 s. The 0.5 s floor turns a normal one-word utterance into ratio 2.3.
  - Short utterances score SECS 0.33. ECAPA2 is being asked to identify a speaker from 0.7–1.0 s of audio, which is too little speech for a reliable embedding. Short-text quality has to be judged with WER, CER, absolute duration and UTMOS, not SECS.
- **The code fingerprint changed during the run** (48 files at setup, 49 at the smoke run): it hashed the reference-voice lock that step 6 creates. It now covers code only; see `eval/harness.get_code_fingerprint`. Both manifests in this run share one fingerprint, so the comparisons inside it are valid.

## Rescored under metrics version 2 (`pilot_en_metrics_v2.json`)

The duration and SECS problems above led to new metric definitions (`eval/metrics.py`, `eval/harness.py`;
`metrics_version` 2). `scripts/rescore_baseline.py` applied them to the utterances measured here, without
re-synthesizing anything. WER, CER, SECS and UTMOS values are unchanged; only derived values and summaries
are recomputed.

- **Duration** is estimated from the words XTTS is asked to say: upstream's own `multilingual_cleaners`
  ("4,800,000" becomes "four million, eight hundred thousand"), not Whisper's normalizer, which turns "six" into "6".
- **Too long** now needs both conditions: more than 2.2× the estimate, *and* at least 1 s over it.
- **Headline SECS** counts only clips of at least 2 s. `secs_all` keeps every clip, and each category is reported separately.

| Category | Duration ratio v1 → v2 | SECS (≥ 2 s) | SECS, all clips | Too long v1 → v2 |
|---|---|---|---|---|
| harvard | 0.98 → 0.98 | 0.559 | 0.559 | 0 → 0 |
| sibilant | 1.14 → 1.07 | 0.481 | 0.481 | 0 → 0 |
| numbers | 1.61 → 0.82 | 0.614 | 0.614 | 0 → 0 |
| long_complex | 0.94 → 0.94 | 0.615 | 0.615 | 0 → 0 |
| prosody_questions | 0.84 → 0.84 | 0.589 | 0.589 | 0 → 0 |
| short | 1.54 → 1.54 | — (n = 0) | 0.332 | 1 → 0 |

Overall: duration ratio 1.17 → 1.03; headline SECS 0.571 ± 0.020 over 100 clips (0.532 over all 120); too-long count 1 → 0.

The 2 s cutoff separates the categories cleanly: every short clip is 0.69–1.15 s, and every other clip is at
least 2 s. The short category's duration ratio (1.54) still sits on the 0.5 s estimate floor; for one-word
text, judge absolute duration instead.
