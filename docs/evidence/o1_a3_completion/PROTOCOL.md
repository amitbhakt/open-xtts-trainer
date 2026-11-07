# O1: finishing ablation A3 (does F2 cause runaway speech?)

**Status: locked before any O1 run.** The lock is commit `d46813b`, which added this file and was pushed before
any run, so its timestamp is public. After the lock, a change to this file, `scripts/run_o1.py` or
`scripts/analyze_o1.py` is an **amendment** if it is made before the first O1 run (listed in section 9) and a
**deviation** if it is made after (section 10). Both are recorded with their reason.

## 1. Why

A3 compared two Stage 2 fine-tunes that differ only in prompt masking: upstream's (defect F2) and per-sample (our
fix). On 300 paired takes it found 4 runaway takes with F2 and 0 with the fix (Fisher p = 0.12), and 9 vs 2 badly
failed short prompts (p = 0.05). Both comparisons were chosen after seeing the data, so they are exploratory
(paper §6.1). A3's pre-registration asked for 300 pairs per seed (900 in all); it ran 300 in all.

O1 does two things:
1. **Confirms or fails to confirm**, on data no one has seen, the hypothesis that A3 suggested: F2 makes runaway
   takes more common.
2. **Completes A3 as pre-registered**: its gates on 900 pairs per arm.

## 2. What is measured

| Item | Value |
|---|---|
| Models | The 6 A3 checkpoints on the `open-xtts-runs` volume (`ablations/a3/{arm}/seed_{1,2,3}/export`). No training. |
| Checkpoint check | Each `model.pth` must match the SHA-256 A3 recorded (`EXPECTED_MODEL_SHA256` in `scripts/run_o1.py`); otherwise the run refuses to start. |
| Arms | `upstream` (F2) and `per_sample` (fix) |
| Voices | The 5 CMU ARCTIC voices: slt, clb, bdl, awb, ksp |
| Sentences | All 60 English sentences of the frozen eval set (SHA-256 `86c0183f…`), in `select_subset` order |
| Generation seed | Equal to the training seed (1, 2, 3), as in A3 |
| Takes | 5 voices × 60 sentences × 3 seeds = 900 per arm, paired by (voice, sentence, seed) |
| Scoring | `scripts/run_eval_baseline.py`: Whisper large-v3, batch 16, metrics version 2, ECAPA2, UTMOS |
| Hardware | One Modal A100-40GB container runs both arms, alternating seed by seed (upstream 1, per-sample 1, upstream 2, …) (amendment A1); the GPU name is recorded in `environment.json` |

**Old and new sentences.** A3 already generated and analysed the first 20 sentences (listed in
`A3_SENTENCE_IDS`, `scripts/analyze_o1.py`; a test checks they are exactly the ones in A3's manifests). The other
**40 are new**: these checkpoints have never generated them. New takes: 40 × 5 × 3 = **600 pairs**.

## 3. Primary (confirmatory) question

**H1: on the 40 new sentences, runaway takes are more common with upstream masking than with per-sample masking.**

- **Runaway take:** WER above 1.0 (`RUNAWAY_WER` in `scripts/compare_ab.py`, unchanged since M4).
- **Test:** exact one-sided McNemar test on the 600 paired takes. Discordant pairs: b = runaway with upstream only,
  c = with per-sample only. p = P(X ≥ b) for X ~ Binomial(b + c, 0.5). Alpha 0.05.
- **Spread check:** sentence-cluster bootstrap of the runaway-rate difference (upstream − per-sample), resampling
  the 40 new sentences with replacement, 10,000 resamples, seed 0; one-sided 95% lower bound (5th percentile).

| Decision | Rule | What the paper says |
|---|---|---|
| **Established** | McNemar p < 0.05 **and** lower bound > 0 | F2 causes runaway takes, across sentences |
| **Supported, concentrated** | McNemar p < 0.05, lower bound ≤ 0 | F2 causes runaway takes, but on one or a few sentences |
| **Not confirmed** | McNemar p ≥ 0.05 | No confirmation at this size. This is **not** evidence that F2 is harmless (section 5) |

Only the 600 new pairs count for this decision. The 300 old pairs generated the hypothesis and are excluded.

## 4. Other outcomes (reported, no decision attached)

1. **Short prompts, new sentences only:** takes with CER above 0.5, same McNemar test (7 new short sentences,
   105 pairs). Secondary, so its p-value is reported without a decision.
2. **Where runaways occur:** counts per sentence, per training seed and per voice, for both arms.
3. **A3 completed:** A3's pre-registered gates (`eval/pre_registered_ablations.json`, unchanged: short-prompt CER
   not worse by 0.022, SECS not worse by 0.02) on all 900 pairs per arm, via `compare_ab` (base = upstream,
   candidate = per-sample), with the robust WER/CER statistics. 300 of these pairs repeat sentences A3 already
   analysed, which the paper states.
4. **Reproduction (exploratory):** the code has changed since A3 (fingerprint `f642df08…` then; this commit now).
   For the 20 old sentences, the share of identical ASR transcripts and the runaway counts, old vs new.

## 5. Power, stated before running

A3's runaways are concentrated: 3 of 4 are on `en_53` ("Seven.") and 1 on `en_51` ("Yes."), both one-word
prompts among the old 20. The 40 new sentences include 7 short prompts but only one single word (`en_54`, "Why?").

Simulated power of the primary test (20,000 draws, upstream runaways ~ Poisson, per-sample ~ Poisson):

| Expected runaways in 600 new upstream takes | Per-sample | Power |
|---|---|---|
| 8 (A3's overall rate, 4 in 300) | 0 | 90% |
| 8 | 0.5 | 79% |
| 4 (half that rate) | 0 | 37% |
| 2 | 0 | 5% |

A3's observed rate is probably an overestimate (it is the result that prompted this test). If F2's runaways are
confined to one-word prompts, the new sentences hold few chances for them, and **"not confirmed" is a likely outcome
even if F2 is harmful.** O1 tests whether the effect generalises to new sentences, not whether it exists on "Seven.".

## 6. Commands

From the repository root, one container for both arms, Mac kept awake for the whole run (`modal run --detach`
does not survive the client shutting down):

```bash
caffeinate -i uvx modal run --detach scripts/modal_run_eval.py --gpu A100-40GB --evidence-dir runs/ablations/o1 --cmd "python scripts/run_o1.py"
```

Then, locally:

```bash
python scripts/analyze_o1.py runs/ablations/o1/manifest_upstream_s60.json runs/ablations/o1/manifest_per_sample_s60.json --report runs/ablations/o1/o1_report.json
```

Results, manifests and the report are copied to this folder. No audio or weights leave Modal.

## 7. Cost and time

A3 synthesised a take in about 2.5 s on this GPU; with scoring, about 1 hour per arm. One A100-40GB container
for both arms: about 2 GPU-hours, **about $5**, and about 1.5–2 hours of wall time. The Modal timeout is 4 hours.

## 8. Conduct

- Every run is reported, including failed or repeated ones. The runner refuses to start if any O1 file already
  exists in its output folder (results or `environment.json`), so it never overwrites or resumes. If a run stops
  early, its folder is kept as it is and the whole run is repeated as a new attempt in a new folder
  (`--output_dir runs/ablations/o1_attempt2`, and `--evidence-dir` to match); the failed attempt is reported.
- Nothing in sections 2–5 changes after a result is seen. The decision is the table in section 3, applied as written.
- The paper and the upstream issue (idiap/coqui-ai-TTS#601) are updated with the result whatever it is.

## 9. Amendments before the first run

| ID | Date | Change | Why |
|---|---|---|---|
| A1 | 2026-11-07 | Both arms run in one container, alternating seed by seed, instead of one container per arm. `run_o1.py` loses `--arm`; one environment record instead of one per arm. | M1 found that the same stock model gives slightly different results in different containers (M1 PROTOCOL.md §10.3, deviation 2: WER changed on 3 of 300 takes, one of them a runaway). With one container per arm, such a difference would fall on one arm only and could create or hide a discordant runaway pair, which the primary test counts. Raised in review of PR #15, before any O1 run. |
| A2 | 2026-11-07 | The no-overwrite check also covers `environment.json`; a stopped run is repeated in full in a new output folder, never resumed (section 8); the lock commit is named and the amendment/deviation rule stated once; wording fixes in `run_o1.py`. | Review of PR #15. No change to what is measured or how it is analysed. |

The design as first locked is commit `d46813b`. No O1 result existed when A1 or A2 was made.

## 10. Deviations

1. **Attempt 1 failed before producing any take; the launch command gained a setup step.** Launched 2026-11-07
   from `9ed01ca` (main after PR #15). The checkpoint check passed for all 6 models, then the first evaluation
   (upstream, seed 1) stopped: the container had no reference voice clips (`eval/reference_voices/arctic_*.wav`
   missing). They are built in each container by `scripts/fetch_reference_voices.py`, which checks them against
   the committed lock (`eval/dataset/reference_voices.lock.json`); the M1 runner calls it, `run_o1.py` does not, and
   the pre-run checks missed it. No take was generated or scored, so no result was seen. Attempt 1's record is in
   `attempt1_failed/` (its `environment.json`: NVIDIA A100-SXM4-40GB; the error); its folder on the runs volume,
   `ablations/o1/`, is left as it is. Attempt 2 runs in a new folder with the command below, which only adds the
   reference-voice step; the code, the clips (same lock) and everything in sections 2–5 are unchanged.

   ```bash
   caffeinate -i uvx modal run --detach scripts/modal_run_eval.py --gpu A100-40GB --evidence-dir runs/ablations/o1_attempt2 --cmd "python scripts/fetch_reference_voices.py && python scripts/run_o1.py --output_dir runs/ablations/o1_attempt2"
   ```

## 11. Results (added after the run; sections 1–10 are unchanged)

Attempt 2 ran on 2026-11-07 from about 17:20 to 18:47 UTC, in one Modal container (NVIDIA A100-SXM4-40GB), on the
locked code (code fingerprint `bf3163b7…` in both manifests) and the 6 checkpoints A3 scored (hashes checked by the
runner). The analysis is `scripts/analyze_o1.py` as locked, run once. All files are in `attempt2/`, unedited;
the report is `attempt2/o1_report.json`.

### 11.1 Primary (section 3): **not confirmed**

| | Upstream (F2) | Per-sample (fix) |
|---|---|---|
| Runaway takes on the 600 new pairs | 1 | 0 |
| Discordant pairs | 1 | 0 |
| Exact one-sided McNemar p | 0.50 | |
| Sentence-cluster bootstrap, one-sided 95% lower bound | 0.000 | |

The one new runaway is `en_56` ("Good morning.", a short prompt), `arctic_awb`, seed 1, WER 1.5. By the rule in
section 3 the decision is **not confirmed**: on sentences A3 never generated, runaway takes were too rare to show a
difference. As section 5 says, this is not evidence that F2 is harmless.

### 11.2 Other outcomes (section 4; no decision attached)

1. **Short prompts, new sentences:** CER above 0.5 in 2 of 105 upstream takes and 0 of 105 per-sample takes
   (McNemar p = 0.25).
2. **Where runaways occur, all 900 pairs:** 5 upstream, 0 per-sample. Four are the takes A3 already reported, on
   the one-word prompts `en_53` ("Seven.", 3) and `en_51` ("Yes.", 1); the fifth is the new `en_56` take. The 900-pair
   counts mix the sentences that generated the hypothesis with new ones, so they are not a test of it.
3. **A3 completed (900 pairs per arm, A3's pre-registered gates): every gate passes.** The fix is not worse than
   upstream masking: WER −0.269 [−0.776, −0.007]; CER −0.310 [−0.844, −0.014]; short-prompt CER −1.856 [−5.125,
   −0.081] (n = 150); SECS −0.0094 [−0.0121, −0.0067] (n = 752, takes of at least 2 s), inside the 0.02 margin;
   UTMOS +0.017 [−0.001, +0.035]; failures 7 vs 15. Mean WER and CER are again dominated by the runaway takes:
   medians are 0 in both arms, and without runaway takes mean WER is 0.023 (upstream) vs 0.019 (per-sample).
   300 of these pairs repeat the sentences A3 already analysed. The fix therefore stays, now with A3's decision made
   on the pre-registered 900 pairs.
4. **Reproduction of A3 (exploratory):** on the 20 sentences A3 scored, 299 of 300 upstream transcripts and 300
   of 300 per-sample transcripts are identical to A3's, under code changed since A3 (`f642df08…` then), and the same
   4 runaway takes recur. The harness reproduces A3's takes almost exactly.

### 11.3 What this means

- F2 is a real bug (its unit test shows it) and the fix is safe: on 900 pre-registered pairs it is not worse on
  any gate.
- The claim that F2 causes runaway speech is **not confirmed** on new sentences. The runaway takes seen so far sit
  almost entirely on two one-word prompts, where they recur deterministically for the same seeds; on 40 new
  sentences they occurred once in 600 takes. The paper should describe F2's measured effect as "rare runaway takes,
  concentrated on one-word prompts, not confirmed in general", not as a general cause of runaway speech.
- Cost: about 1.5 GPU-hours on one A100-40GB (about $3.5), plus a few minutes for attempt 1.
