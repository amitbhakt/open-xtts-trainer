# M1: a Stage 2 setting that adapts the voice without costing other voices (protocol)

**Status: FROZEN 2026-10-31 (`78ce960`), AMENDED 2026-10-31 – 2026-11-03 before any M1 training run (section 9).** This is the
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
  75 clips, as many as the target's training clips (50% of C2's training clips). The target training set must have
  exactly 75 clips (as produced for Moira Fogarty) to match the fixed 75-clip replay set and implement the
  registered 50% replay ratio. The runner enforces this: runs where the target training set does not have 75 clips
  are rejected under `--replay-librittsr`. Text is the corpus's normalized transcript.
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
  than reference words, a heuristic for run-on or looping takes, not proof) with its Clopper–Pearson interval, median WER/CER, the mean without runaway takes; paired, the median difference, the paired mean difference over pairs where neither arm ran away, and the
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
  baselines are regenerated per run). Screening is four runs; each confirmation attempt costs two runs (seeds 2 and 3;
  even if seed 2 fails, seed 3 still runs to complete the candidate's three-seed stability record); C4 one
  more, plus its two confirmation runs if needed. Total runs increase by two per failed confirmation attempt.
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

## 9. Amendments before the first run (2026-10-31 – 2026-11-03)

Made after `78ce960` froze the protocol and before any M1 training run, prompted by reviews of PR #14. None uses
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
| A10 | C2 runner enforces exactly 75 target training clips (`--replay-librittsr`) | C2 fixes replay at 75 clips to match the target's 75 clips (50% replay ratio); target splits with another count would alter the registered ratio |
| A11 | Budget and confirmation: 2 runs per confirmation attempt, and seed 3 still runs if seed 2 fails | A failed confirmation also costs runs; completing seed 3 records full seed-stability data for that candidate |

## 10. Results (added after the runs; sections 1–9 are unchanged)

**Status: M1 FAILS (rule 6).** C0–C3 ran on 2026-11-06 and C4 on 2026-11-07, on Modal (A100-SXM4-40GB). No
candidate passed both gate sets, in screening or in the second round, so nothing was confirmed. Every file the
runner returned is in `c0/`, `c1/`, `c2/`, `c3/` and `c4/` beside this file, unedited.

### 10.1 Screening results (training seed 1)

| Run | Target gates (`gates_target.json`) | Preservation gates (`gates.json`) | Pass |
|---|---|---|---|
| C0 | All pass. SECS +0.0329 [+0.0191, +0.0477]; WER +0.0084; CER +0.0033; UTMOS −0.040; failures 0 and 0 | **SECS fails**: −0.0243 [−0.0299, −0.0185] against a 0.02 margin. WER −0.0365, CER −0.1318, UTMOS +0.097, failures (tuned 2, stock 6) pass | No |
| C1 | All pass. SECS +0.0400 [+0.0263, +0.0539]; WER −0.0046; CER +0.0016; UTMOS −0.040; failures 0 and 0 | SECS passes: −0.0113 [−0.0164, −0.0062]. **WER fails**: +0.0163 [−0.027, +0.080]. **CER fails**: +0.0373 [−0.142, +0.291]. UTMOS +0.065 and failures (tuned 5, stock 6) pass | No |
| C2 | SECS +0.0442 [+0.0293, +0.0592]; WER −0.0017; UTMOS +0.021; failures 0 and 0 pass. **CER fails**: +0.0037 [−0.0033, +0.0143] | **SECS fails**: −0.0322 [−0.0383, −0.0260]. **CER fails**: −0.0793 [−0.199, +0.010]. WER −0.0204, UTMOS +0.132, failures (tuned 3, stock 6) pass | No |
| C3 | All pass. SECS +0.0266 [+0.0147, +0.0398]; WER −0.0058; CER +0.0013; UTMOS −0.021; failures 0 and 0 | **SECS fails**: −0.0189 [−0.0238, −0.0140]. WER −0.0171, CER −0.0898, UTMOS +0.061, failures (tuned 3, stock 6) pass | No |
| C4 (second round) | All pass. SECS +0.0480 [+0.0338, +0.0631]; WER −0.0058; CER +0.0013; UTMOS −0.019; failures 0 and 0 | SECS passes: −0.0127 [−0.0177, −0.0076]. **WER fails**: +0.0193 [−0.0225, +0.0676]. **CER fails**: +0.0827 [−0.113, +0.319]. UTMOS +0.056 and failures (tuned 6, stock 6) pass | No |

Brackets are 95% CIs of the tuned − stock difference, as `compare_ab.py` reports them; a `not_worse` gate fails
when the CI's worse end crosses the margin, which is why C3 fails at a mean of −0.0189. C0 reproduces Run 6's
preservation failure (Run 6: SECS −0.0236 [−0.029, −0.018]).

**What the SECS gates cover.** Headline SECS pairs only takes at least 2 s long in both arms
(`HEADLINE_SECS_MIN_S`, `eval/harness.py`), because ECAPA2 is unreliable on shorter speech. On the preservation
set that keeps 253–256 of the 300 pairs, and almost no short prompts (0–2 of 45 per run). So the preservation SECS
gate says nothing about forgetting on short utterances; those are covered only by the WER, CER and failure gates.

**Data versions.** C1, C2 and C3 ran on the second of the two data versions (section 10.3, deviation 1), so their
comparisons with C0 and C4 carry that deviation. No candidate passed, so the verdict is the same whether or not
they are counted.

### 10.2 Secondary outcomes (section 5; never gates)

| | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|
| Best step, eval loss before → best | 190, 3.958 → 3.105 | 190, 3.954 → 3.125 | 380, 3.954 → 3.084 | 190, 3.954 → 3.272 | 380, 3.958 → 3.099 |
| Training clips (target + replay), eval clips, LR | 75, 8, 5e-6 | 75, 8, 5e-6 | 75 + 75 (15 readers), 8, 5e-6 | 75, 8, 2.5e-6 | 75 + 75 (15 readers), 8, 5e-6 |
| Frozen parameter tensors | 0 | 54 (`conditioning`) | 0 | 0 | 54 (`conditioning`) |
| Preservation SECS diff by voice: awb, bdl, clb, ksp, slt | −0.028, −0.008, −0.012, −0.030, −0.044 | −0.018, −0.003, −0.002, −0.010, −0.022 | −0.030, −0.014, −0.020, −0.064, −0.033 | −0.012, −0.004, −0.019, −0.020, −0.039 | −0.018, +0.000, −0.009, −0.017, −0.020 |
| Preservation runaway takes (WER > 1), stock / tuned | 3 / 0 | 2 / 2 | 2 / 0 | 2 / 1 | 3 / 4 |
| Preservation median WER difference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Preservation paired mean WER difference without runaway pairs | −0.0133 | −0.0037 | −0.0105 | −0.0105 | −0.0075 |
| Preservation paired mean CER difference without runaway pairs | −0.0321 | −0.0162 | −0.0127 | −0.0267 | −0.0109 |
| Target runaway takes, stock / tuned; target median CER difference | 0 / 0; 0.0 | 0 / 0; 0.0 | 0 / 0; 0.0 | 0 / 0; 0.0 | 0 / 0; 0.0 |

Every runaway take in all five runs, in either arm, is on sentence `en_53` ("Seven.", a one-word item). C1's
tuned arm has two: `arctic_awb`, seed 123, at WER 8.0, and `arctic_ksp`, seed 999, at WER 2.0. The first alone
adds 0.027 to the 300-take mean WER, more than C1's whole WER difference (+0.0163). C2's target CER failure comes from one take:
the tongue twister `en_12`, seed 123 ("The six sick sheiks' sixth sheep is sick."), at CER 0.393 in the tuned arm
and 0.107 in the stock arm, a paired difference of 0.286; on its own that pair adds 0.0048 to the 60-take mean
difference, more than C2's whole CER difference (+0.0037). Both are reported, not used: the gates are unchanged, and C1 and C2 fail them. C2's replay made
preservation SECS worse than C0's, not better (−0.032 against −0.024), most of all on `arctic_ksp` (−0.064).
C4's preservation WER failure comes entirely from `en_53`: that sentence adds +0.0233 to the 300-take mean WER
difference (+0.0193); the other 295 pairs together give −0.0040. Its tuned arm has four runaway takes on it, three
on `arctic_bdl` (seeds 42, 123, 999), against three in the stock arm. Reported, not used: C4 fails the gates.

### 10.3 Deviations and observations

1. **Training data are not byte-identical across runs (section 2, "Same data for every run").** C0 and C1's
   `data_identity.json` differ in `train_dataset_sha256` and `train_clips_sha256`; the held-out reference
   digest matches. Only two versions occurred: C0, C4, the refused first C2 attempt and the cancelled first C4
   attempt share one (`e8b609a8…`); C1, C2 and C3 share the other (`9634d52e…`). All runs selected the same 83
   clips (75 training, 8 eval, the same eval clip names). The difference comes from re-preparing the target data
   in each run: forced alignment on the GPU moved some cut boundaries by about 0.01–0.1 ms and some word scores
   by 0.01 (`prep_report.json`). By section 2's rule this is
   a deviation, reported here. C1 is still compared, since the clip set is the same and only the sample-level
   boundaries differ; a reader who rejects that comparison should treat C1, C2 and C3 (the runs on the second
   version) as not compared with C0. The verdict does not depend on it: no candidate passed either way.
2. **Stock baselines regenerated per run are not bit-identical.** Same checkpoint (`model.pth` `c7ea2000…`),
   eval set (`86c0183f…`), reference voices, seeds and batch size (16). Every take's SECS differs slightly between
   C0's and C1's stock runs; WER differs on 3 of 300 preservation takes, one of them a runaway (`arctic_ksp`,
   seed 123, `en_53`: WER 3.0 in C0's stock run, 1.0 in C1's). Stock preservation mean WER is 0.0756 in C0 and
   0.0697 in C1. C1's, C2's and C3's stock runs are bit-identical take by take (all 60 target and 300
   preservation takes), and C4's equal C0's take by take; the two groups match the two data versions in item 1,
   so the container environment, not the run's options, decides which of two near-identical outcomes a run
   gets. Each A/B compares against its own run's stock arm, as section 2 prescribes.
3. **The recorded git tree does not identify the commit (amendment A4).** Both runs record commit `ba28fe2` and
   tree `7348db7`, which match no commit in the repository (the commit is the container's own, and its tree
   holds more than the repository). The code fingerprint does identify the code: `b0882b43…` (70 files), equal to
   the fingerprint of `8dd17ed` (main after PR #14), recomputed from that commit's files. So both runs ran the
   frozen, amended protocol code. C3 records the same fingerprint, `b0882b43…`. C2 and C4 record `60f25919…`, the
   fingerprint of the fixed code (item 5).
4. The copy of `gates_target.json` in each results directory (all five runs) differs from `eval/gates_target.json` only by a
   missing final newline; the parsed contents are identical, and both reports record gates version 1.
5. **Code change after freezing: C2's target-clip check (section 8, amendment A10).** On 2026-11-06 at 18:48 UTC
   the first C2 run stopped before training: `--replay-librittsr` refused to train because
   `data_moira_fogarty/dataset_train.jsonl` "has 83" clips, not 75. The check, added in A10 (`403538a`), counted
   the lines of that file, which still hold the 8 eval clips the trainer splits off; the protocol's "75 target
   training clips" are the clips left after that split (C0 and C1: 83 in the file, 75 trained on, 8 eval). So the
   check refused every correctly prepared C2 run. It had no test. The fix (`scripts/run_target_finetune_ab.py`,
   `check_c2_target_clips`) counts the clips the trainer trains on, using the trainer's own `split_samples` with
   the same eval fraction (0.1) and split seed (1), and still refuses any other count. Nothing else changes:
   training, the replay set and its checks, evaluation and the gates are the same code, and the training command is
   unchanged. Two tests cover it: the real 83-clip shape passes and 75, 82 and 84 lines are refused (the old check
   fails this test), and the split reproduces C0's and C1's recorded 75/8. Run on the real training file from the
   Modal volume (byte-identical to C1's, by `data_identity.json`'s digest), the fixed check gives 75. The fix was
   made, tested and committed before any C2 or C3 result was known; C3 was still training on the frozen code. C2
   therefore runs on code whose fingerprint differs from C0's, C1's and C3's (`b0882b43…`), by this check alone
   (see the commit diff); the fixed code's fingerprint is `60f25919…` (70 files). Confirmation runs, if any, also run on the fixed code. The failed C2 attempt's partial evidence showed
   one more thing: its `data_identity.json` equals C0's exactly, so re-preparation sometimes gives byte-identical
   data (item 1). That attempt's evidence and its refusal message are kept in `c2_attempt1_refused/`.
6. **Launch history.** C2 and C3 were first launched at about 18:44 UTC without `--detach` and stopped by us within
   about a minute, before any training, to relaunch them detached.
   The only files those launches wrote, one `environment.json` per run on the runs volume, were deleted before
   relaunching. C3's relaunch is the C3 run reported here.
7. **C4's first attempt was cancelled; C4 was run again in full.** Launched 2026-11-06 at 20:29 UTC (detached), it
   trained (150 training clips, 75 of them replay) and was cancelled at 20:50:59 UTC during evaluation, when the
   local machine shut down: the runner's local client waits on the remote call and cancels it when told to quit,
   which `--detach` does not prevent (`c4_attempt1_cancelled/cancellation.txt`). It produced no A/B report. Its
   evidence files are kept in `c4_attempt1_cancelled/`; its two partial target manifests were not opened, not
   used and not committed. Its results folder and its trained checkpoint were deleted from the runs volume, and C4
   was run again in full on 2026-11-07 (launched 03:43 UTC, finished in 34.9 minutes) with the same code
   (`60f25919…`) and command, the local machine kept awake. That rerun is the C4 reported here. The replay set was
   built faster than in C2 because C2 had left the archive cached on the runs volume; the runner still refused to
   train unless the set matched the committed selection, and it did.
8. **Two evidence files need reading with care (noted after review; the files are unchanged).**
   `evidence/data_identity.json`'s `train_clips` (83) is the number of lines in `dataset_train.jsonl`, eval clips
   included. `training_summary.json` records the clips actually trained on: 75 target clips for C0, C1 and C3, and
   150 for C2 and C4 (75 target + 75 replay, also given as `same_language_replay_clips`), the same distinction as
   item 5. `evidence/identity_canary.xml` is `test_identity_canary_released_model`, run in each container on
   the **released** XTTS-v2 weights: it checks each run's environment, not the candidate's checkpoint, and is not
   evidence about preservation.

### 10.4 Where the decision rule stands (section 4)

The screening round is complete and no candidate passed (C0, C1, C2, C3). Rule 5 applied: C4 (C1 + C2:
`--freeze conditioning --replay-librittsr`, `--subset-size 20`, `--run-tag c4`) was screened next, on the fixed
code, to be confirmed with training seeds 2 and 3 if it passed. This decision followed the rule as written and was
recorded before C4 was run.

**C4 does not pass** (section 10.1), so no candidate can be confirmed and **M1 fails (rule 6)**. As section 4
prescribes, the recipe states that Stage 2 adaptation on this speaker costs zero-shot cloning of other voices, and
every run's numbers are reported above. No further M1 candidate is run, and the gates are not changed.

What the runs show, without changing the verdict: every candidate passes the target-speaker SECS gate (+0.027 to
+0.048). Preservation SECS, the forgetting measure, passes only when the conditioning path is frozen (C1 −0.011,
C4 −0.013; otherwise −0.019 to −0.032). C1 and C4 fail only the preservation WER and CER gates, and in both the
failure comes from runaway takes on the one-word sentence `en_53`. Those two gates also fail when the stock model
is compared with itself at other seeds (section 10.5). These observations come after the results and are
exploratory; they can motivate a new, separately pre-registered protocol (for example M2 with the conditioning
path frozen, and gates checked against stock-vs-stock noise before freezing), not a change to M1's verdict.

### 10.5 Exploratory: do the gates pass the stock model against itself? (not used to choose)

Run on 2026-11-06 at about 20:10 UTC, on stock results only, before C2's preservation results were returned
(20:16 UTC). No gate, verdict or decision changes because of it; it is reported to help read the verdicts.
`compare_ab.py`'s own gate code was used unchanged, except that its refusal to compare a checkpoint with itself
was bypassed in a scratch script.

- **Same seeds** (C0, C1 and C3's stock runs, three pairs, both gate sets): 0 of 27 `not_worse` gates fail. These
  takes are nearly or exactly identical, so this shows only that rerunning the stock model does not, by itself,
  fail a gate.
- **Different seeds** (each of C0's and C1's stock runs compared with itself, its seeds relabelled 42→123→999→42
  and 42→999→123→42, so each take is paired with the same voice and sentence at another seed; eight comparisons):
  every target gate passes (CER interval about ±0.005, WER ±0.008, UTMOS ±0.034), and preservation SECS passes
  (about ±0.005 against a 0.02 margin). **Preservation WER and CER fail in all four preservation comparisons**:
  their intervals are about ±0.03 (WER, margin 0.02) and ±0.11 (CER, margin 0.01), driven by runaway takes.

Reading: at this test size, a model whose outputs differ from stock only by sampling chance can fail the
preservation WER and CER gates, so a pass or failure on those two gates says little about the model; C0's passes
and C1's failures there are consistent with chance. The SECS gate and the target gates are much wider than their
noise, so their failures (C0 −0.024, C2 −0.032, C3 −0.019 on preservation SECS) are not explained by chance.
Seed relabelling approximates the noise between two different models; it is not a measured false-failure rate.
For M2, gates should be checked this way before they are frozen. (Added after C4: C4's preservation WER and CER
failures are on these same two gates; this check was run before C4, and C4 does not change it.)
