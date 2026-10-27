# Recipe design: what the three stages do, why, and how sure we are

This document explains the training recipe. For each choice it gives the evidence behind it and the test
that guards it. Code comments refer here with short IDs:
- **F1–F7:** problems found in the upstream code (§2);
- **§3:** recipe decisions;
- **A1–A8:** experiments still to run (§5);
- **Phase 0–9:** verification stages (§6).

**Upstream** means [`idiap/coqui-ai-TTS`](https://github.com/idiap/coqui-ai-TTS), the maintained fork of
[`coqui-ai/TTS`](https://github.com/coqui-ai/TTS) (unmaintained since Coqui shut down; last push 2024-08-16, not
formally archived). It is pinned at commit `03339e5b790b5e16ea8867a142e8bb26ec646d94` (`pyproject.toml`), and every
file:line reference below is to that commit unless marked "original". The code findings were also checked against
the original's last release, v0.22.0, where each one is present too (table below).

## §1 Evidence levels

No claim is accepted because of who made it. A decision rises in level only through a test or a measured
experiment (§5).

| Tag | Meaning | How to treat it |
|---|---|---|
| `[E1]` | Verified in the pinned upstream code, or measured in this repository | Build on it; guard it with a test |
| `[E2]` | Peer-reviewed result (noted when it is a single study or a different domain) | Strong default; still confirm on our setup |
| `[E3]` | Preprint or adjacent domain (arXiv only, language models, ASR) | Reasonable default; confirm by experiment |
| `[E4]` | Practitioner report: uncontrolled, often one team and one dataset | A hypothesis with some support; experiment required |
| `[E5]` | Our hypothesis: inferred, not yet tested | Test before it shapes a default |

## §2 Problems found in the upstream code

Each finding has a test that demonstrates the upstream behaviour, and a guard that fails if our fix is removed.

| ID | Finding | Where | Our fix | Tests |
|---|---|---|---|---|
| **F1** | The DVAE codebook is not learned by gradient. In train mode, `Quantize.forward` overwrites it on every forward pass with a running average that keeps 99% of the old value. Any DVAE training changes what the codes mean, whatever the learning rate, so a trained DVAE no longer matches the pretrained GPT. | `tts/layers/xtts/dvae.py:103-116` | Stage 1 is off by default. A trained DVAE is marked `requires_gpt_retrain`; Stage 2 refuses it with pretrained GPT weights unless told to retrain, and latent extraction refuses a GPT not trained on it. Validation runs in eval mode. | `test_codebook_collapse_fails_the_run`, `test_stage1_dvae_with_pretrained_gpt_is_refused`, `test_dvae_pairing_check`, `test_extract_latents_refuses_mismatched_dvae` |
| **F2** | GPT training removes the voice-prompt region from the loss only for the **last sample of each batch**: the masking line sits after the per-sample loop and reuses its index. With the perceiver resampler (the released config), the prompt span is also divided by 256 (conditioning-mel frames) instead of 1024 (codes), so even that sample has the wrong codes masked. Every other sample is trained to predict audio it was just given as its prompt. | `tts/layers/xtts/gpt.py:461-465` | The mel loss is recomputed from upstream's own logits with every sample's prompt masked in code units. | `test_upstream_f2_masks_only_the_last_sample_in_the_wrong_units`, `test_prompt_masking_per_sample`, `test_gpt_loss_parity_without_prompt` |
| **F3** | The released `dvae.pth` has 53 tensors; the model has 56. The three missing ones are `discrete_loss.accumulator*` running-statistics buffers. Upstream and community trainers load with `strict=False`, which hides this and any real mismatch. | the file's own index | Strict loading with an exact allowlist of those three buffers. | `test_dvae_strict_load_allows_only_unsaved_buffers`, `test_upstream_dvae_loads` |
| **F4** | The stock tokenizer raises `NotImplementedError` for any language outside its list, and inference indexes `char_limits[language]`, so a model trained on a new language can't run through stock inference. | `tts/layers/xtts/tokenizer.py:651-652` | A tokenizer subclass with explicitly registered languages, used in validation, training, the exported inference loader and latent extraction. | `test_project_tokenizer_registers_languages`, `test_train_export_and_new_language_inference` |
| **F5** | Hugging Face Whisper word timestamps are contiguous: each word ends exactly where the next begins, so they can't locate pauses. Also: when a transcript is missing words, CTC forced alignment doesn't leave the extra speech unaligned; it stretches a neighbouring word over it. | `tests/fixtures/jfk_whisper_alignment.json` | Word times come from CTC forced alignment (MMS_FA). Missing text is detected as frames where the model hears a character but the alignment path says blank. | `test_whisper_word_timestamps_have_no_gaps`, `test_missing_text_is_detected_as_unexplained_speech` |
| **F6** | When `GPTTrainer` extends the vocabulary, new text rows get unit-variance random values, and the old *last* row is copied onto the new last row on the assumption that it is `[START]`. In XTTS-v2 `[START]` is 261 and the last row is `[hi]` (6680), so the newest token, typically the new language tag, starts as a copy of `[hi]`. | `tts/layers/xtts/trainer/gpt_trainer.py:113-142` | New rows start at the mean of the existing rows plus small noise (A5). Base weights load strictly through our own path. | `test_new_text_rows_mean_init_and_upstream_f6` |
| **F7** | The fork's `Xtts._load_checkpoint` catches the `UnpicklingError` a legacy checkpoint raises, converts its pickled `XttsConfig` to a dict, and `torch.save`s the result **over the original `model.pth`**. The released file stores its config that way, so the first load rewrites it: its SHA-256 changed from the locked `c7ea2000…` to `3077a05b…` in Colab runs 1–2, and those runs' baseline manifests can't be tied to the locked weights. Not in coqui-ai/TTS v0.22.0: the fork introduced it. | `TTS/tts/models/xtts.py:639-652` | `patch_xtts_prevent_checkpoint_mutation()` (`xtts_trainer/__init__.py`) does the conversion in memory; every loader applies it. | `test_upstream_f7_load_rewrites_a_legacy_checkpoint_in_place`, `test_guarded_load_leaves_the_checkpoint_untouched`, `test_every_xtts_checkpoint_load_applies_the_guard` |

**Also in the original `coqui-ai/TTS` v0.22.0** (tag commit `fa28f99f`, checked 2026-10-24; commands, file hashes
and the exact lines are in [`docs/evidence/upstream/coqui_tts_v0.22.0_check.md`](evidence/upstream/coqui_tts_v0.22.0_check.md)):
F1, F2, F4 and F6 date from Coqui's own release, not from the fork. F7 is the exception: the fork introduced it,
and v0.22.0's loader has no such rewrite (same evidence file). F3 is a property of the released `dvae.pth`
itself and F5 of Hugging Face Whisper, so neither depends on the code base. Paths in both columns are relative to
the repository root of each project.

| Finding | Original v0.22.0 | Fork, pinned commit |
|---|---|---|
| F1 | `TTS/tts/layers/xtts/dvae.py:103-116` | `TTS/tts/layers/xtts/dvae.py:103-116` |
| F2 | `TTS/tts/layers/xtts/gpt.py:541-546` (mask after the loop), `:414` (perceiver divides by 256) | `TTS/tts/layers/xtts/gpt.py:461-465` |
| F4 | `TTS/tts/layers/xtts/tokenizer.py:643` | `TTS/tts/layers/xtts/tokenizer.py:651-652` |
| F6 | `TTS/tts/layers/xtts/trainer/gpt_trainer.py:118-137` | `TTS/tts/layers/xtts/trainer/gpt_trainer.py:113-142` |
| F7 | absent: `TTS/tts/models/xtts.py` has no `UnpicklingError` handler and no `torch.save` | `TTS/tts/models/xtts.py:639-652` (fork only) |
| Silent sample substitution | `TTS/tts/layers/xtts/trainer/dataset.py:156-180` | same pattern |
| Non-strict checkpoint and DVAE loads | `TTS/tts/layers/xtts/trainer/gpt_trainer.py:84, 186` | same |
| Half-clip prompt for short clips | `TTS/tts/layers/xtts/trainer/dataset.py:35` | `TTS/tts/layers/xtts/trainer/dataset.py:36-37` |

One difference: the original cleans Hindi with `basic_cleaners` (a "will implement" placeholder); the fork uses
`multilingual_cleaners`, like the other listed languages.

Other upstream behaviours the Stage 2 wrapper changes:
- **Silent sample substitution:** `XTTSDataset.__getitem__` catches every exception and silently trains on sample 1 instead (and can recurse forever). Ours raises; `test_dataset_failures_raise_instead_of_substituting`, `test_strict_dataset_items_match_upstream`.
- **Unsafe defaults:** `GPTArgs` defaults to an 8194-token codebook and no perceiver, which don't match the released model. The architecture comes from the released config instead; `test_config_architecture_comes_from_the_released_model`.
- **Lost epoch:** in the coqui `Trainer`, `start_with_eval=True` turns the first epoch into an evaluation, so N epochs train N−1. Not used.

From the XTTS paper ([arXiv 2406.04904](https://arxiv.org/abs/2406.04904), Interspeech 2024), `[E2]`:
- **DVAE:** the codebook was trained with 8192 codes, then pruned to the 1024 most used, at 21.53 Hz.
- **Text:** the GPT uses a 6681-token BPE vocabulary; Chinese, Japanese and Korean are romanized first.
- **Decoder:** the HiFi-GAN decoder is trained on GPT latents with a Speaker Consistency Loss, and outputs 24 kHz.
- **Fine-tuning:** about 10 minutes of data per speaker raised cross-lingual SECS from 0.585 to 0.717 (no hyperparameters given).
- **Evaluation:** CER from Whisper large-v3, SECS from ECAPA2, UTMOS.
- **Not stated:** the VQ-VAE's training data, how training references are chosen, the decoder's full loss, and any procedure for adding languages.

## §3 Recipe decisions

"Built" means implemented and tested here; "Planned" means not built yet.

**Data (all stages)**

| Decision | Evidence | Status |
|---|---|---|
| Transcripts come from the corpus's own metadata or from forced alignment, never from fixed-length cuts. Each clip's text is exactly its aligned words. | `[E1]` | Built (`xtts_trainer/prepare/`); real-audio check passed (`test_mms_jfk_alignment_and_shifted_cut`) |
| Reject a clip when re-alignment finds a word below 0.05, more than three words below 0.2, or more than 0.1 s of unexplained speech. | `[E5]` calibrated on 8 baseline recordings (clean native clips score $\ge 0.54$, accented `arctic_ksp` scores 0.062 with 3 weak words; a 1 s cut shift drops to 0.01–0.02); held-out run (pre-registered, locked environment): clean acceptance validated out-of-sample `[E1]` for English (59/60), French/German too few clips to validate; detection reliable only for large corruptions (head cuts ≥ 300 ms, 600 ms tail cut, inserted opening word, 0.5 s spillover) | Built; calibrated on 8 baseline speakers `[E5]`; validated out-of-sample on unseen speakers and languages in locked environment (66/67 clean accepted, 98.5%; `docs/evidence/phase6_heldout/`) `[E1]` |
| Cut only at pauses of at least 150 ms, preferring the longest pauses. | `[E4]` practitioners; `[E1]` for the long-pause preference (a cut at a tight boundary lost a clip on real JFK audio) | Built; `test_cuts_prefer_the_longest_pause_jfk` |
| Mix short utterances and single words in with long ones from the start, not in a later round. | `[E4]`; `[E2]`-adjacent: forgetting under sequential fine-tuning ([Yang et al. 2021](https://www.arxiv.org/pdf/2110.04482v1)) | The validator warns when under 5% of clips are short; thresholds are A1 |
| Punctuation should match pauses (a comma with no pause, or a long pause with no punctuation, is flagged). | `[E4]` | Built as a warning; `--strict_pauses` makes it blocking; A6 decides the default |
| Minimum clip length: upstream's 0.5 s. | `[E1]` upstream; `[E4]` reports that 0.3 s helps single words | 0.5 s built; 0.3 s is part of A1 |

**Stage 1: DVAE**

| Decision | Evidence | Status |
|---|---|---|
| Off by default: fine-tune the GPT (and optionally the decoder) with the stock DVAE. | `[E1]` F1; `[E4]` community practice | Built (pairing rules, F1) |
| Train it only when a diagnostic shows the stock DVAE fits your audio clearly worse than reference speech (code usage, reconstruction), and then retrain Stage 2. | `[E1]` F1; `[E5]` thresholds | Diagnostic built (`scripts/diagnose_dvae.py`); thresholds are A2 |
| Never swap in a DVAE whose codes differ (e.g. Tortoise's) under a pretrained GPT. | `[E1]` + `[E2]` (codebook pruned 8192→1024) | Enforced by the pairing rules |
| Frozen-codebook mode (quantizer fixed, encoder trainable). | `[E1]` that it doesn't protect the GPT | Built, but it doesn't keep codes stable: on real speech 81% of codes changed, the same as normal mode (synthetic data: 47%), because the encoder adapts ([`colab_run4`](evidence/colab_run4/README.md)). Any Stage 1 run needs a Stage 2 retrain; A2 |
| Fail the run if the codebook collapses on validation data. | `[E1]` + `[E2]` ([Zheng & Vedaldi, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Online_Clustered_Codebook_ICCV_2023_paper.html)) | Built; floor values are `[E5]` |

**Stage 2: GPT and text**

| Decision | Evidence | Status |
|---|---|---|
| A thin wrapper over upstream `GPTTrainer`; don't reimplement the loop. | `[E1]`; `[E4]` community success with it | Built (`stage2_gpt/gpt_trainer.py`) |
| Mask each sample's own prompt region (F2). | `[E1]` | Built; whether it improves quality is A3 |
| Prompt from the same clip (upstream default, masking fixed); offer a different clip of the same speaker as an option. | `[E1]` default; `[E4]`/`[E5]` alternative | Default built; the option is planned (A4) |
| Keep `min_conditioning_length` at 3 s. | `[E4]` conflicting reports | Default; A1/A4 |
| New text rows start at the mean of existing rows plus small noise (F6). | `[E3]`/`[E2]`-adjacent: [CoNLL 2024](https://aclanthology.org/2024.conll-1.8.pdf), [FOCUS](https://arxiv.org/pdf/2305.14481), [WECHSEL](https://arxiv.org/pdf/2112.06598); language-model studies, none on TTS | Built; A5 |
| New-language text: per-character tokens now; BPE merges or romanization as alternatives. | Per-character `[E1]` safe; BPE `[E4]`; romanization `[E2]` (the XTTS paper romanizes CJK), `[E3]` ([arXiv 2601.05776](https://www.arxiv.org/pdf/2601.05776)) | Per-character built; alternatives are A5 |
| A tokenizer subclass with registered languages, in training and inference (F4). | `[E1]` | Built |
| New-language runs mix in base-language replay data; evaluation covers base languages. | `[E2]` ([Yang et al. 2021](https://www.arxiv.org/pdf/2110.04482v1)); `[E3]` same effect in ASR and LLMs | Built (replay must be other languages; shares are reported); ratio is A7 |
| Starting hyperparameters: AdamW (β 0.9, 0.96), weight decay 1e-2 on weights only, LR 5e-6, effective batch ≥ 32, MultiStepLR. | `[E4]` upstream recipe defaults | Defaults, not truths; tune against eval loss |

**Stage 3: HiFi-GAN decoder**

| Decision | Evidence | Status |
|---|---|---|
| Optional: run it after a large GPT change, and keep the result only if it beats the stock decoder in the A/B. | `[E2]` ([HiFi-GAN](https://arxiv.org/pdf/2010.05646), single study on mel features); `[E4]` conflicting reports | Built; worth is A8 |
| Extract latents with the GPT in eval mode, teacher-forced with the true transcript, from the GPT you will run. | `[E1]` (dropout; eval-mode latent trimming) | Built; `test_vocoder_round_trip_prefers_true_alignment` on real weights |
| Interpolate the decoder input over the whole clip, then cut segments. | `[E1]` (`HifiDecoder.forward` interpolates linearly) | Built; exact alignment, `test_stage3_segments_align_with_decoder` |
| Speaker Consistency Loss as an option. | `[E2]` used for the XTTS decoder, but the YourTTS erratum says its gradient never flowed there ([arXiv 2112.02418](https://arxiv.org/pdf/2112.02418)) | Built, off by default; A8 |
| Full-band (to 12 kHz) mel loss as an option; the default stops at 8 kHz. | `[E4]` | Built, off by default; A8 |
| Discriminators start fresh, so warm them up while the generator learns from mel loss only; refuse runs that never leave warm-up. | `[E1]` (no upstream discriminator weights exist); `[E4]` noisy early epochs | Built |
| Keep the best checkpoint by validation mel loss, not the last one. | `[E1]` smoke run: mel L1 1.553 → 1.350 at step 106, then back up to 1.367 by step 120 ([`colab_run4`](evidence/colab_run4/README.md)) | Built |

**Evaluation**

| Decision | Evidence | Status |
|---|---|---|
| CER/WER from Whisper large-v3 with the standard normalizer; speaker similarity from ECAPA2 (not the model's own encoder); UTMOS only relative to the baseline. | `[E2]` XTTS paper protocol; `[E3]` ([arXiv 2506.20190](https://arxiv.org/pdf/2506.20190)) | Built (`eval/`) |
| Duration is estimated from the words XTTS actually says (upstream's text cleaner); "too long" needs both a 2.2× ratio and ≥ 1 s excess; headline speaker similarity counts clips ≥ 2 s. | `[E1]` measured: the written-digit and sub-second-clip artifacts in the stock baseline | Built (metrics version 2) |
| Fine-tunes are judged by a paired A/B against stock XTTS-v2, with gates fixed in advance. | `[E1]` | Built (`scripts/compare_ab.py`, `eval/gates.json`) |
| Voice fine-tunes are also judged on the target speaker, with reference clips held out of training (the validator refuses them), not only on unseen eval voices. | `[E1]` (a fine-tune can't be credited for cloning a voice it trained on) | Built (`scripts/build_target_voices.py`, `--voices target`) |
| Training data has a stated license, and every downloaded file is checked against a SHA-256 before use (no pin, no download). | `[E1]` | Built. LJSpeech and JFK: commit-pinned URLs. Art of War: archive.org URLs aren't revision-specific, so the hash is the guarantee (changed content fails the check). Each SHA-256 was taken from a file whose SHA-1 matched the original pin (archive.org's published value, or the value computed at pinning), so provenance carries over without relying on SHA-1. Commands and values in [`docs/evidence/source_pins.md`](evidence/source_pins.md). |

## §4 Community claims, assessed

Sources: [anhnh2002](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages),
[daswer123 dvae-finetune](https://github.com/daswer123/xtts-finetune-tests/tree/main/dvae-finetune),
[tuanh123789](https://github.com/tuanh123789/Train_Hifigan_XTTS), and the discussion in
[coqui-ai/TTS#3704](https://github.com/coqui-ai/TTS/issues/3704).

| Claim | Verdict | Basis |
|---|---|---|
| "The DVAE is universal; never train it" | Right default, wrong reason | Coupling (F1) `[E1]`; universality unproven |
| "You can use the pretrained Tortoise DVAE" | Wrong for fine-tuning | Codebook pruned 8192→1024 `[E2]`; code ids change |
| "The DVAE decoder isn't used at inference" | Right | The decoder runs on GPT latents `[E2]` |
| "An LM head recomputes the mel from codes" | Wrong | The GPT head predicts DVAE codes `[E1]` |
| "Short-text hallucination is purely a data problem" | Partly right | Recipe mechanics are also implicated (F2, half-clip prompts, the 0.5 s filter) `[E1]`/`[E5]`; no controlled test exists |
| "Mix short and long data; don't add short data later" | Likely right | Consistent with the forgetting literature `[E2]`; one report `[E4]` |
| "Lowering `min_conditioning_length` fixes short text" / "it breaks cloning" | Both unproven | No measurements `[E4]` |
| "Commas must match pauses" | Plausible, low risk | `[E4]` |
| "Use a different clip of the same speaker as the reference" | Unproven | `[E5]`; no head-to-head comparison |
| "Decoder fine-tuning made it worse" / "promising at 80 h" / "unnecessary" | All conditionally right | Depends on data size, SCL and setup `[E2]`/`[E4]` |
| "No DVAE training needed with ~20 h of data" | Plausible | `[E4]`; consistent with F1 |

## §5 Experiments still to run (A1–A8)

Each runs through the evaluation harness against stock XTTS-v2: ≥ 3 seeds, confidence intervals, and the
decision rule fixed before running. Results update the evidence tags above.

| ID | Question | Setup | Primary metric | Decision rule |
|---|---|---|---|---|
| A1 | What fixes short-text hallucination? | Share of short clips (0/5/15%) × minimum clip length (0.5/0.3 s) × `min_conditioning_length` (3/1 s) | CER on 1–3-word sentences; SECS on held-out speakers | Cheapest setting that brings short-text CER within the long-text CER's interval without lowering SECS beyond the gate |
| A2 | When is training the DVAE worth it? | Stock vs trained vs frozen-codebook DVAE, each with a full Stage 2 retrain | Code usage, reconstruction, then end-to-end CER/SECS/UTMOS | Recommend Stage 1 only where end-to-end metrics improve beyond the interval |
| A3 | Does fixing F2 matter? | Upstream masking vs per-sample masking | Short-text CER, SECS, duration failures | Keep the fix (it is correct either way); report the effect |
| A4 | Same-clip vs different-clip prompt | Masked same-clip vs same-speaker different-clip prompt | SECS on held-out speakers; CER | Default to the better one |
| A5 | Text for a new script, and new-row initialization | BPE merges vs per-character vs romanization; mean vs random new rows | CER; tokens per sentence; no base-language regression | Best CER that passes the base-language check |
| A6 | Does punctuation/pause consistency matter? | Same data with and without commas added at long pauses (`--fix_pauses`) | Pause placement, CER, UTMOS | Make the check blocking if the effect exceeds the interval |
| A7 | Replay ratio for new languages | 0/10/25% base-language replay | Target CER; base-language CER/SECS | Smallest ratio with no base-language regression beyond the gate |
| A8 | Is Stage 3 worth it? | Stock vs fine-tuned decoder (± SCL, ± full-band loss) after a new-language GPT run | UTMOS, SECS, CER | Keep Stage 3 only if it beats the stock decoder beyond the interval |

## §6 Verification phases

The phase names used in code comments and in `notebooks/phase0_phase1_colab.ipynb`. Evidence is in
`docs/evidence/`. Status as of 2026-10-26.

| Phase | Checks | Status |
|---|---|---|
| 0 Pin and record | Upstream pinned; every artifact fetched and hash-verified by one script; `mel_stats` identical across three sources | Done (Colab runs 1–4; Run 5 re-verified every artifact after each step it used them) |
| 1 Baseline | Stock XTTS-v2 measured on a frozen eval set; reruns identical | Done for same-environment reruns; across environments, reruns are not bit-identical, and means agree within intervals except where single runaway takes dominate: Spanish WER moved 0.099 → 0.213 from one take, inside a tolerance too wide to fail, and pooled agreement doesn't rule out a language-specific regression (`docs/evidence/phase8_reproducibility/README.md`). Same environment: bit-identical reruns (English pilot, runs 1–2). Run 5: 5-language baseline on the locked weights, 900 utterances (`eval/baselines/base_xtts_s12_index.json`). Cross-environment rerun (Phase 8, different commit and GPU session): 852/900 takes produced identical audio; 48 differ, and one Spanish take became a repetition loop that doubles Spanish mean WER (0.099 → 0.213; medians 0.0 in both). Not bit-identical across environments |
| 2 Single source of truth | One spec module, one mel front end, one model factory, strict loading | Done |
| 3 Data contracts | One dataset format, validated before every stage; latent files carry a schema | Done |
| 4 Parity | Our spectrograms, models and losses match upstream's, each check shown to fail when broken | Done |
| 5 Canaries | Vocoder round trip, export identity, overfit, segment alignment | Done; identity on the released weights verified (Run 6, `test_identity_canary_released_model` in `docs/evidence/run6_target_speaker/evidence/identity_canary.xml`) |
| 6 Data preparation | Transcribe, force-align, cut at pauses; a cut moved by 1 s must fail validation | Done: real-aligner checks (runs 3–4); sample set prepared and validated with zero edits (run 4) |
| 7 Training | Stage 2 on upstream `GPTTrainer` with the fixes above; Stages 1 and 3 with validation, resume and best-by-validation checkpoints | Smoke runs on real weights pass all three stages (run 4); target speaker fine-tune completed (Moira Fogarty *Art of War*, 10 epochs on A100, loss 3.947 → 3.102) |
| 8 Evaluation | Paired A/B against stock with pre-registered gates; README numbers only from manifests | Run 6, one fine-tune measured both ways. Target speaker ($N=60$, `eval/gates_target.json`, the rules `eval/gates.json` fixed in `603e0d3`): all 5 gates pass, SECS +0.0329 [+0.019, +0.048]. Preservation on 5 unseen voices ($N=300$, `eval/gates.json`): **fails** its SECS gate, −0.0236 [−0.029, −0.018] against a 0.02 margin. Manifests in `docs/evidence/run6_target_speaker/` |
| 9 Automation | CI, pre-commit, PR checklist, fresh-install check | CI: done. On every push it installs from a clean checkout with `uv sync --frozen --extra dev`, runs ruff, and runs the test suite with `vocab.json`, `config.json` and `mel_stats.pth` only, so tests needing the full weights, MMS_FA or the reference voices skip there and run on GPU (Colab/Modal). Pre-commit: `.pre-commit-config.yaml` (ruff and the static-invariant tests) is in place; whether contributors run it isn't enforced. PR checklist: `.github/pull_request_template.md` is in place. Fresh-install script: `scripts/check_fresh_install.sh` exists, but CI does not run it (run it by hand) |

## §7 Bug-class protocol (for every change)

1. **Reproduce:** write a test that fails on the current code.
2. **Classify:** name the class ("spectrogram settings differ from upstream"), not just the instance.
3. **Search siblings:** grep for every instance of the class; record what you searched for and found.
4. **Fix at the source of truth:** change the one module that owns the fact.
5. **Verify external facts:** numbers or text from outside the repo (hashes, timestamps, transcripts, language support) come from running a command and saving its output, never typed in.
6. **Prove it:** the new test passes, and a deliberately broken version of the fix makes it fail.
7. **Measure:** if model behaviour can change, attach harness numbers before and after.
8. **Close:** a finding is closed only when it is linked to the test that guards it.

Thresholds come from the baseline, a published reference or a stated requirement, never from the current
output, with the reason written next to the number. The same rules apply to changes written with an AI assistant.

## §8 Sources

Papers:
- [XTTS (Casanova et al., Interspeech 2024)](https://arxiv.org/abs/2406.04904)
- [HiFi-GAN (Kong et al., NeurIPS 2020)](https://arxiv.org/pdf/2010.05646)
- [YourTTS and erratum](https://arxiv.org/pdf/2112.02418)
- [Lifelong multilingual TTS (Yang et al. 2021)](https://www.arxiv.org/pdf/2110.04482v1)
- [Online clustered codebook (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Online_Clustered_Codebook_ICCV_2023_paper.html)
- [Embedding initialization (CoNLL 2024)](https://aclanthology.org/2024.conll-1.8.pdf)
- [FOCUS (EMNLP 2023)](https://arxiv.org/pdf/2305.14481)
- [WECHSEL](https://arxiv.org/pdf/2112.06598)
- [Romanization in LM pretraining](https://www.arxiv.org/pdf/2601.05776)
- [ECAPA vs H/ASP in zero-shot TTS](https://arxiv.org/pdf/2506.20190)
- [Speaker-embedding leakage](https://ar5iv.org/html/2010.11221)
- [Forced alignment with MMS (Pratap et al. 2023)](https://arxiv.org/abs/2305.13516)
