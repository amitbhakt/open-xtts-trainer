# M1: a Stage 2 setting that adapts the voice without costing other voices (protocol)

**Status: DRAFT, not yet pre-registered.** This file becomes the pre-registration when its status line reads
"FROZEN" in a commit made before the first M1 training run. Anything changed after that commit is reported as a
deviation. Results go in a separate section added after the runs; nothing above it changes.

## 1. Question

Run 6 fine-tuned the Stage 2 GPT on 6.4 minutes of one target speaker. It passed all five target-speaker gates
(SECS +0.033, 95% CI [+0.019, +0.048]) but failed the preservation gate on the five unseen eval voices: SECS
−0.0236, CI [−0.029, −0.018], against a 0.02 margin. The recipe can't be recommended while adaptation costs
zero-shot cloning of other speakers.

**M1 asks:** is there a Stage 2 setting that passes **both** gate sets, unchanged, on the Run 6 target speaker?

## 2. What stays fixed (identical to Run 6)

| Item | Value |
|---|---|
| Target speaker and data | Moira Fogarty, *The Art of War* (LibriVox, public domain), prepared as in Run 6: 75 training clips, 8 eval clips, held-out reference clips in `eval/target_voices/` (private by design; the dataset validator refuses training on them while their lock exists) |
| Eval split | The target's own seeded split (`--eval_fraction 0.1`). Same-language replay never enters it (`stage2_gpt/train.py`), so every candidate's best checkpoint is chosen on the same 8 target clips |
| Base model | Released XTTS-v2, hash-verified against `artifacts.lock.json` |
| Training defaults | 10 epochs, batch 4 × 8 accumulation (effective 32), AdamW, per-sample prompt masking (F2 fix), best checkpoint by eval loss |
| Target evaluation | `eval/gates_target.json` (version 1): SECS `better` 0.0; WER `not_worse` 0.02; CER 0.01; UTMOS 0.10; failures 0. 20 sentences (`--subset-size 20`) × eval seeds 42, 123, 999 (N = 60) |
| Preservation evaluation | `eval/gates.json` (version 1): WER `not_worse` 0.02; CER 0.01; UTMOS 0.10; SECS 0.02; failures 0. Five ARCTIC eval voices × 20 sentences × 3 eval seeds (N = 300), against the locked stock manifest `eval/baselines/base_xtts_s20_en.json` |
| Environment | Locked (`uv sync --frozen`), Modal A100-40GB as Run 6; every run records its commit |

Neither gate file changes. A candidate is judged by `scripts/compare_ab.py` exactly as Run 6 was.

## 3. Candidates (screening round, training seed 1)

Each is one change from Run 6, so a pass can be attributed.

| ID | Change | Why | Runner options |
|---|---|---|---|
| C0 | None: Run 6 retrained with the current code | Control. Shows whether Run 6's preservation failure reproduces before anything is credited with fixing it | (defaults) |
| C1 | Freeze the speaker-conditioning path (`conditioning_encoder`, `conditioning_perceiver`) | That path maps any reference clip to the 32 latents every voice is cloned from. Training on one speaker can bend it towards that speaker. The rest of the GPT still adapts | `--freeze conditioning` |
| C2 | Same-language replay: other English speakers' clips, as many as the target's (50% of training clips) | Rehearsal against forgetting (Yang et al. 2021, for languages; here for speakers) | `--replay <dataset.jsonl>` |
| C3 | Half the learning rate (2.5 × 10⁻⁶) | Less drift from the released weights over the same steps | `--lr 2.5e-6` |

Each candidate gets its own `--run-tag` and `--results-dir`, so no run overwrites another.

**Second round, only if no screening candidate passes both gate sets:** C4 = C1 + C2. No other combination is run.

**Replay data for C2 (decision needed before freezing; see section 7).** Requirements, whichever source is chosen:
- other speakers than the target, and none of the five ARCTIC eval voices (`slt`, `clb`, `bdl`, `awb`, `ksp`);
- no sentence from the frozen eval set (the dataset validator already refuses these);
- every download pinned by SHA-256, license recorded, prepared and validated like any training data;
- 75 clips from at least 10 speakers, chosen by a seeded rule written down before the download.

## 4. Decision rule (fixed in advance)

1. **Pass** = every gate in `eval/gates_target.json` **and** every gate in `eval/gates.json` passes.
2. Among candidates that pass, choose the one with the largest target SECS gain (point estimate). If two are
   within 0.005, prefer the simpler, in this order: C1, C3, C2, C4.
3. **Confirmation (needed for M1 to count as done):** the chosen setting is retrained with training seeds 2 and 3
   (same data, same eval) and must pass both gate sets at both seeds. Screening four candidates on the same gates
   gives more than one chance to pass by luck; confirmation on fresh training runs is the guard against that.
   These two runs are also this speaker's three seeds for M2.
4. If nothing passes (after C4), M1 fails. The paper then reports the trade-off and every candidate's numbers, and
   the recipe states that Stage 2 adaptation costs zero-shot cloning by the measured amount.

Every run is reported, whatever it shows, including C0 and any failed confirmation.

## 5. Reported for every run (secondary; never gates)

- From the `robust` section of each A/B report (M4): per arm, the runaway rate (takes with WER > 1.0) with its
  Clopper–Pearson interval, median WER/CER, the mean without runaway takes; paired, the median difference and the
  runaway takes that occur in one arm only.
- Per-voice SECS differences for the five preservation voices.
- Eval loss before and at the best step, the best step, and `training_summary.json` (frozen parts, replay share
  and speakers, LR, seed).

## 6. Exploratory diagnostic (not a candidate; not used to choose)

Run 6's fine-tuned weights with only the conditioning path restored to the released values, evaluated on both
gate sets. It needs no training. If preservation recovers and the target gain stays, the conditioning path is
where forgetting happens; if not, it isn't. It is labelled exploratory because it is chosen with knowledge of
Run 6's result and could suggest settings after the fact. It needs a small checkpoint-merging script (not yet
written).

## 7. Open before freezing

- **Replay source for C2.** Two options:
  - *Other CMU ARCTIC speakers* (not the five eval voices, nor `rms`/`jmk` used by the alignment validation). Cheap:
    the project already reads and pins ARCTIC tarballs. But ARCTIC is the preservation voices' own recording
    setup, so replaying it could raise preservation scores through recording-condition familiarity rather than
    through speaker generality. That would bias the result towards passing.
  - *A different multi-speaker corpus* (e.g. LibriTTS-R). No shared recording setup with the preservation voices,
    so a pass is more convincing. Needs a pinned downloader and prep; its license must be read and recorded at
    fetch time.
  - Recommendation: a different corpus.
- **Budget.** Screening is four training runs and four evaluations of 360 takes each; confirmation two more of
  each; C4 one more if needed.
- **Second preservation voice set (for M2).** All candidates are chosen on the same five ARCTIC voices, so the
  winner is partly fitted to them. M2 should add preservation voices unused during selection.

## 8. Code this protocol relies on

- `xtts_trainer/stage2_gpt/train.py`: same-language replay (other speakers only, training only) and `--freeze`;
  `training_summary.json` records both. Tested in `tests/test_stage2_gpt_trainer.py`.
- `scripts/run_target_finetune_ab.py`: `--train-seed`, `--freeze`, `--replay`, `--run-tag`; at the defaults the
  training command is Run 6's. Tested in `tests/test_target_voices.py`.
- `scripts/compare_ab.py`: the gates, and the M4 `robust` section.
