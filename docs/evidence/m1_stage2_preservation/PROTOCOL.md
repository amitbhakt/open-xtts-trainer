# M1: a Stage 2 setting that adapts the voice without costing other voices (protocol)

**Status: FROZEN 2026-10-31, before any M1 training run.** This is the pre-registration: the commit that first
shows this status line is the reference. Anything above the results section changed after that commit, and any
change to the code in section 8, is reported as a deviation. Every M1 run records the commit it ran from. Results
go in a separate section added after the runs; nothing above it changes.

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
| C2 | Same-language replay: other English speakers' clips, as many as the target's (50% of training clips) | Rehearsal against forgetting (Yang et al. 2021, for languages; here for speakers) | `--replay-librittsr` |
| C3 | Half the learning rate (2.5 × 10⁻⁶) | Less drift from the released weights over the same steps | `--lr 2.5e-6` |

Each candidate gets its own `--run-tag` and `--results-dir`, so no run overwrites another.

**C2 also doubles the training steps (decided 2026-10-31).** With 75 replay clips added to 75 target clips and the
same 10 epochs, C2 takes twice as many optimizer steps as C0, while every target clip is still seen the same number
of times. C2 is defined as replay *at equal target exposure*, not at equal steps. Matching steps instead (5 epochs)
would halve how often the target is seen, which changes something else. So a C2 pass is attributed to "replay plus
the extra steps it brings", and the paper says so.

**Second round, only if no screening candidate passes both gate sets:** C4 = C1 + C2. No other combination is run.

**Replay data for C2: LibriTTS-R `dev-clean`** (decided 2026-10-31). A restored multi-speaker audiobook corpus at
24 kHz (Koizumi et al. 2023; https://www.openslr.org/141/), licensed CC BY 4.0 by Google LLC (its `LICENSE.txt`).
It shares no recording setup with the five ARCTIC preservation voices, so a pass can't come from familiarity with
their channel. Built by `scripts/build_replay_librittsr.py`:
- **Archives:** `dev_clean.tar.gz` and `doc.tar.gz`, each checked against the MD5 OpenSLR publishes in its
  `md5sum.txt` (an independent check, not trust-on-first-use) and a SHA-256 pin.
- **Speakers:** the 40 `dev-clean` readers (20 F, 20 M, from `speakers.tsv`), minus excluded readers. Female and
  male ids are each sorted and shuffled with seed 20261031 into a queue; 15 slots are filled alternately (female
  first), 8 F and 7 M, each from the front of its gender's queue. A reader without 5 usable clips is skipped and
  recorded, and the next reader of the same gender takes the slot (LibriTTS-R drops noisy lines, so a few readers
  have little audio).
- **Excluded:** reader 209, Moira Fogarty, the target speaker. She is in LibriTTS-R (`train-clean-360`), not in
  `dev-clean`; the exclusion is explicit so the rule holds whichever subset is used.
- **Clips:** per reader, utterances sorted and shuffled with a per-reader seed; the first 5 that last 3–10 s and
  pass the project's dataset validator (vocabulary, frozen eval-set sentences, reference-voice audio, durations).
  75 clips, as many as the target's training clips (50% of C2's training clips). Text is the corpus's normalized
  transcript.
- **Recorded:** `replay_selection.json` lists the pins, license, rule, every chosen clip (with its SHA-256),
  skipped readers and every rejected candidate with its reason.
- **Built 2026-10-31** (selection uses only the corpus, no training result): 75 clips, 7.5 min, from readers 1988,
  5536, 2035, 251, 1993, 777, 6345, 2803, 2412, 6295, 8842, 7976, 84, 5694, 6313. Reader 3576 was skipped (15
  utterances, 4 of them 3–10 s long). All 217 rejected candidates failed the duration rule; none failed validation.
  A second build gave a byte-identical `replay_selection.json`.

XTTS-v2's own training data may include LibriTTS-derived audiobooks; we have not verified this and claim nothing
either way. For replay it would not be a flaw: rehearsing data the model was trained on is what replay is meant
to do.

LibriTTS-R comes from LibriVox, like the target recording. That makes replay resemble the target's channel, not
the preservation voices', so if anything it makes the preservation gate harder to pass with C2, not easier.

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

## 7. Decisions and notes recorded at freezing

- **Replay source for C2:** decided, LibriTTS-R `dev-clean` (section 3). Other CMU ARCTIC speakers were rejected:
  they share the preservation voices' recording setup, which could raise preservation scores through channel
  familiarity rather than speaker generality.
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
- `scripts/build_replay_librittsr.py`: C2's replay set. Tested on fake archives in
  `tests/test_build_replay_librittsr.py`.
- `scripts/run_target_finetune_ab.py --replay-librittsr`: builds the replay set in the run's own environment, with
  the 1.36 GB archive cached under `runs/_cache/librittsr` (a persistent Modal volume, so it downloads once), and
  refuses to train unless the build matches the committed `replay_selection.json` exactly, every clip's SHA-256
  included. Tested in `tests/test_target_voices.py`.
