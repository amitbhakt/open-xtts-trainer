# M1: a Stage 2 setting that adapts the voice without costing other voices (protocol)

**Status: FROZEN 2026-10-31 (`78ce960`), AMENDED 2026-10-31 before any M1 training run (section 9).** This is the
pre-registration. Its reference is the amended version; section 9 lists every change from `78ce960` and why. Any
later change above the results section, or to the code in section 8, is reported as a deviation. Results go in a
separate section added after the runs; nothing above it changes.

## 1. Question

Run 6 fine-tuned the Stage 2 GPT on 6.4 minutes of one target speaker. It passed all five target-speaker gates
(SECS +0.033, 95% CI [+0.019, +0.048]) but failed the preservation gate on the five unseen eval voices: SECS
−0.0236, CI [−0.029, −0.018], against a 0.02 margin. The recipe can't be recommended while adaptation costs
zero-shot cloning of other speakers.

**M1 asks:** is there a Stage 2 setting that passes **both** gate sets, unchanged, on the Run 6 target speaker?

## 2. What stays fixed

| Item | Value |
|---|---|
| Target speaker and data | Moira Fogarty, *The Art of War* (LibriVox, public domain), prepared by the current pipeline. Run 6 had 75 training and 8 eval clips; the alignment thresholds have changed since (0.10 s unexplained-speech limit), so the counts may differ and are recorded per run. Held-out reference clips are in `eval/target_voices/` (private by design; the dataset validator refuses training on them while their lock exists) |
| Same data for every run | Every run writes `evidence/data_identity.json`: SHA-256 digests of the training dataset, of every training clip, and of the held-out reference folder. All runs must match C0's; a run that doesn't is a deviation and is reported, not compared |
| Eval split | The target's own split, seeded with split seed 1 (Run 6's) in every run, including confirmation runs at other training seeds (`--split_seed 1`, passed by the runner). Same-language replay never enters it. So every run chooses its best checkpoint on the same target clips; `training_summary.json` lists them and their speakers |
| Base model | Released XTTS-v2, hash-verified against `artifacts.lock.json` |
| Training defaults | 10 epochs, batch 4 × 8 accumulation (effective 32), AdamW, per-sample prompt masking (F2 fix), best checkpoint by eval loss |
| Target evaluation | `eval/gates_target.json` (version 1): SECS `better` 0.0; WER `not_worse` 0.02; CER 0.01; UTMOS 0.10; failures 0. 20 sentences (`--subset-size 20`) × eval seeds 42, 123, 999 (N = 60) |
| Preservation evaluation | `eval/gates.json` (version 1): WER `not_worse` 0.02; CER 0.01; UTMOS 0.10; SECS 0.02; failures 0. Five ARCTIC eval voices × 20 sentences × 3 eval seeds (N = 300) |
| Stock baselines | Regenerated in every run by the runner, for both evaluations, with batched evaluation (batch 16) like the tuned model. The locked `eval/baselines/base_xtts_s20_en.json` records no batch size, so the runner does not reuse it, and `compare_ab.py` refuses to compare manifests evaluated at different batch sizes. Run 6 was evaluated one take at a time, so C0 repeats Run 6's training under today's evaluation, not Run 6 exactly |
| Environment | Locked (`uv sync --frozen`), Modal A100-40GB as Run 6. Modal uploads no `.git`, so the commit a run records is the container's own; each run also records the git tree hash and the code fingerprint, which identify the real commit |
| Every run's options | `--subset-size 20` (the runner's default is 12), plus the candidate's options in section 3 |

Neither gate file changes. A candidate is judged by `scripts/compare_ab.py` with the same gates as Run 6.

## 3. Candidates (screening round, training seed 1)

Each is one change from Run 6, so a pass can be attributed.

| ID | Change | Why | Runner options |
|---|---|---|---|
| C0 | None: Run 6's training, with the current code | Control. Shows whether Run 6's preservation failure reproduces before anything is credited with fixing it | none beyond `--subset-size 20` |
| C1 | Freeze the speaker-conditioning path (`conditioning_encoder`, `conditioning_perceiver`) | That path maps any reference clip to the 32 latents every voice is cloned from. Training on one speaker can bend it towards that speaker. The rest of the GPT still adapts | `--freeze conditioning` |
| C2 | Same-language replay: other English speakers' clips, as many as the target's (50% of training clips) | Rehearsal against forgetting (Yang et al. 2021, for languages; here for speakers) | `--replay-librittsr` |
| C3 | Half the learning rate (2.5 × 10⁻⁶) | Less drift from the released weights over the same steps | `--lr 2.5e-6` |

Each candidate gets its own `--run-tag`; the runner derives a separate results directory from it, so no run
overwrites another's evidence or reports.

**C2 also doubles the training steps (decided 2026-10-31).** With 75 replay clips added to 75 target clips and the
same 10 epochs, C2 takes twice as many optimizer steps as C0. Upstream draws training clips at random with
replacement (a language, then a clip within it), so no run sees each clip a fixed number of times; C2 keeps the
*expected* number of target draws equal to C0's. C2 is defined as replay *at equal expected target exposure*, not
at equal steps. Matching steps instead (5 epochs)
would halve how often the target is seen, which changes something else. So a C2 pass is attributed to "replay plus
the extra steps it brings", and the paper says so.

**Second round, only when no candidate has been confirmed (section 4):** C4 = C1 + C2. No other combination is run.

**Replay data for C2: LibriTTS-R `dev-clean`** (decided 2026-10-31). A restored multi-speaker audiobook corpus at
24 kHz (Koizumi et al. 2023; https://www.openslr.org/141/), licensed CC BY 4.0 by Google LLC (its `LICENSE.txt`).
It shares no recording setup with the five ARCTIC preservation voices, so a pass can't come from familiarity with
their channel. Built by `scripts/build_replay_librittsr.py`:
- **Archives:** `dev_clean.tar.gz` and `doc.tar.gz`, each checked against the MD5 OpenSLR publishes in its
  `md5sum.txt` (an independent check, not trust-on-first-use) and a SHA-256 pin.
- **Speakers:** the 40 `dev-clean` readers (20 F, 20 M, from `speakers.tsv`), minus excluded readers. Female and
  male ids are each sorted, then shuffled in turn (female first) by one random generator seeded with 20261031,
  giving one queue per gender; 15 slots are filled alternately (female first), 8 F and 7 M, each from the front of
  its gender's queue. A reader without 5 usable clips is skipped and
  recorded, and the next reader of the same gender takes the slot (LibriTTS-R drops noisy lines, so a few readers
  have little audio).
- **Excluded:** reader 209, Moira Fogarty, the target speaker. She is in LibriTTS-R (`train-clean-360`), not in
  `dev-clean`; the exclusion is explicit so the rule holds whichever subset is used.
- **Clips:** per reader, utterances sorted and shuffled with a per-reader seed; the first 5 that last 3–10 s and
  pass the project's dataset validator (vocabulary, frozen eval-set sentences, reference-voice audio, durations),
  looking through the reader's utterances until 5 pass or none are left.
  75 clips, as many as the target's training clips (50% of C2's training clips). Text is the corpus's normalized
  transcript.
- **Recorded:** `replay_selection.json` lists the pins, license, rule, every chosen clip (with its SHA-256),
  skipped readers, and every utterance looked at and not chosen, with its reason (including a skipped reader's
  usable clips).
- **Built 2026-10-31** (selection uses only the corpus, no training result): 75 clips, 7.5 min, from readers 1988,
  5536, 2035, 251, 1993, 777, 6345, 2803, 2412, 6295, 8842, 7976, 84, 5694, 6313. Reader 3576 was skipped (15
  utterances, 4 of them 3–10 s long; those 4 are recorded). Of 361 utterances looked at and not chosen, 217 failed
  the duration rule, 140 were usable but not needed, and 4 belonged to the skipped reader; none failed validation.
  Two builds gave a byte-identical `replay_selection.json`.

XTTS-v2's own training data may include LibriTTS-derived audiobooks; we have not verified this and claim nothing
either way. For replay it would not be a flaw: rehearsing data the model was trained on is what replay is meant
to do.

LibriTTS-R comes from LibriVox, like the target recording. That makes replay resemble the target's channel, not
the preservation voices', so if anything it makes the preservation gate harder to pass with C2, not easier.

## 4. Decision rule (fixed in advance)

1. **Pass** = every gate in `eval/gates_target.json` **and** every gate in `eval/gates.json` passes.
2. **Ranking.** Candidates that pass are ranked by target SECS gain (point estimate), largest first. Two within
   0.005 are ordered by simplicity: C0, C1, C3, C2, C4. C0 is eligible: if the unchanged recipe passes and is
   confirmed, the answer is that no change is needed and Run 6's failure was a bad draw.
3. **Confirmation (needed for M1 to count as done):** the top-ranked setting is retrained with training seeds 2
   and 3 (same data, same split seed 1, same eval) and must pass both gate sets at both seeds. Screening several
   candidates on the same gates gives more than one chance to pass by luck; confirmation on fresh training runs
   is the guard against that. These runs are also this speaker's three seeds for M2.
4. **If confirmation fails,** the next passing candidate in the ranking is confirmed the same way, and so on.
5. **If no candidate is confirmed** and C4 has not run, C4 is screened and, if it passes, confirmed.
6. **Otherwise M1 fails.** The paper reports the trade-off and every run's numbers, and the recipe states that
   Stage 2 adaptation costs zero-shot cloning by the measured amount.

Every run is reported, whatever it shows, including C0 and any failed confirmation.

## 5. Reported for every run (secondary; never gates)

- From the `robust` section of each A/B report (M4): per arm, the runaway rate (takes with WER > 1.0: more errors
  than reference words, a heuristic for run-on or looping takes, not proof) with its Clopper–Pearson interval, median WER/CER, the mean without runaway takes; paired, the median difference and the
  runaway takes that occur in one arm only.
- Per-voice SECS differences for the five preservation voices.
- Eval loss before and at the best step, the best step, and `training_summary.json` (frozen parts, replay share
  and speakers, LR, epochs, batch size, accumulation, eval fraction, training and split seeds, eval clips and their
  speakers).
- `data_identity.json` and the environment record (git tree hash, code fingerprint).

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
- **Budget.** Each run trains once and evaluates 720 takes (stock and tuned, target and preservation, since the
  baselines are regenerated per run). Screening is four runs; confirmation two per candidate confirmed; C4 one
  more, plus its confirmation, if needed.
- **Second preservation voice set (for M2).** All candidates are chosen on the same five ARCTIC voices, so the
  winner is partly fitted to them. M2 should add preservation voices unused during selection.

## 8. Code this protocol relies on

- `xtts_trainer/stage2_gpt/train.py`: same-language replay (other speakers only, training only) and `--freeze`;
  `training_summary.json` records both. Tested in `tests/test_stage2_gpt_trainer.py`.
- `scripts/run_target_finetune_ab.py`: `--train-seed` (other seeds keep split seed 1), `--freeze`, `--replay`,
  `--run-tag` (also separates the results directory); `data_identity.json` and the git tree hash; at the defaults the
  training command is Run 6's. Tested in `tests/test_target_voices.py`.
- `scripts/compare_ab.py`: the gates, and the M4 `robust` section.
- `scripts/build_replay_librittsr.py`: C2's replay set. Tested on fake archives in
  `tests/test_build_replay_librittsr.py`.
- `scripts/run_target_finetune_ab.py --replay-librittsr`: builds the replay set in the run's own environment, with
  the 1.36 GB archive cached under `runs/_cache/librittsr` (a persistent Modal volume, so it downloads once), and
  refuses to train unless the build matches the committed `replay_selection.json` exactly, every clip's SHA-256
  included; it also refuses any other `--replay` file beside it. Tested in `tests/test_target_voices.py`.
- `scripts/modal_run_eval.py --evidence-dir runs/m1/<candidate>`: brings back the run's reports and records, never
  the held-out clips, audio or weights. Tested in `tests/test_modal_run_eval.py`.

## 9. Amendments before the first run (2026-10-31)

Made after `78ce960` froze the protocol and before any M1 training run, prompted by a review of PR #14. None uses
any M1 result; there is none yet.

| # | What changed | Why |
|---|---|---|
| A1 | Stock baselines are regenerated per run at batch 16, not taken from the locked manifest | The runner never reused it (no batch size recorded) and `compare_ab.py` refuses mismatched batch sizes; the frozen text described something that could not happen |
| A2 | The split seed is fixed at 1 in every run, separately from the training seed | Before, `--train-seed` also reseeded the split, so confirmation runs would have chosen checkpoints on different clips while the text said "same eval" |
| A3 | Every run records `data_identity.json`; runs whose target data differ from C0's are deviations | Each run re-prepares the target data, and the pipeline changed after Run 6, so "prepared as in Run 6: 75 and 8 clips" was an assumption, not a fact |
| A4 | The run's real commit is identified by the git tree hash and code fingerprint | On Modal the recorded commit is the container's own; "every run records its commit" was not true as written |
| A5 | C2's exposure is "equal expected target draws" | Upstream samples with replacement; "every target clip seen the same number of times" was false |
| A6 | C0 is eligible and first in the simplicity order; a failed confirmation falls through to the next passing candidate, then to C4 | The rule left both cases undefined |
| A7 | Each run's results directory follows its `--run-tag`; `--replay-librittsr` refuses other replay files | Candidates could overwrite each other's reports, and C2 could have trained on more than the pre-registered set |
| A8 | Replay selection: the reader shuffle is described as it is coded (one generator, female queue first); readers are scanned until 5 clips pass; every utterance looked at and not chosen is recorded | The text described a per-gender seed the code never used; a cap could have skipped a reader with usable clips later in its order; a skipped reader's usable clips went unrecorded. The 75 chosen clips are unchanged (no clip ever failed validation) |
| A9 | Runaway takes are described as a heuristic; every run uses `--subset-size 20`; the budget is 720 takes per run | WER > 1 can't prove a loop; C0 was listed as "(defaults)" though the default subset is 12; the baselines are regenerated per run |
