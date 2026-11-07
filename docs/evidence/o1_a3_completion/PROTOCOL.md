# O1: finishing ablation A3 (does F2 cause runaway speech?)

**Status: locked before any O1 run.** The lock is the commit that adds this file; it is pushed before the first
run, so its timestamp is public. Any later change to this file, `scripts/run_o1.py` or `scripts/analyze_o1.py` is
a deviation and is recorded in section 9 with its reason.

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
| Hardware | Modal A100-40GB for both arms (one container per arm); the GPU name is recorded in `environment_{arm}.json` |

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

From the repository root, one container per arm, both on the same GPU type, Mac kept awake for the whole run
(`modal run --detach` does not survive the client shutting down):

```bash
caffeinate -i uvx modal run --detach scripts/modal_run_eval.py --gpu A100-40GB --evidence-dir runs/ablations/o1 --cmd "python scripts/run_o1.py --arm upstream"
```

```bash
caffeinate -i uvx modal run --detach scripts/modal_run_eval.py --gpu A100-40GB --evidence-dir runs/ablations/o1 --cmd "python scripts/run_o1.py --arm per_sample"
```

Then, locally:

```bash
python scripts/analyze_o1.py runs/ablations/o1/manifest_upstream_s60.json runs/ablations/o1/manifest_per_sample_s60.json --report runs/ablations/o1/o1_report.json
```

Results, manifests and the report are copied to this folder. No audio or weights leave Modal.

## 7. Cost and time

A3 synthesised a take in about 2.5 s on this GPU; with scoring, about 1 hour per arm. Two A100-40GB containers in
parallel: about 2 GPU-hours, **about $5**, about 1 hour of wall time. The Modal timeout is 4 hours.

## 8. Conduct

- Every run is reported, including failed or repeated ones. A rerun is a new attempt with its own record; the
  runner never overwrites a finished arm.
- Nothing in sections 2–5 changes after a result is seen. The decision is the table in section 3, applied as written.
- The paper and the upstream issue (idiap/coqui-ai-TTS#601) are updated with the result whatever it is.

## 9. Deviations

None yet.
