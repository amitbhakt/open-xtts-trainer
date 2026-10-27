# Auditing and Repairing the XTTS-v2 Fine-Tuning Pipeline: An Evidence-Graded, Reproducible Three-Stage Recipe

**Amit Bhakt**¹

¹ *[Affiliation to be added]* · Correspondence: *[e-mail to be added]*

> **Draft status (2026-10-25).** This is a working draft. Sections 1–4 describe the method and are close to
> final. Section 5 reports what has been measured so far: the code audit, the stock-model baseline, the
> data-preparation checks, smoke-scale training runs, and target speaker voice adaptation (§5.6). Full language
> adaptation across non-Latin scripts has not been evaluated yet, so the claims in Section 6 that depend on
> full language fine-tuning remain empirical hypotheses; the gates that will accept or reject this are
> pre-registered in Section 6. Every number in this draft is traceable to a file in this repository
> (Appendix A). Text in *[brackets]* is a placeholder.

---

## Abstract

XTTS-v2 is a widely used open multilingual zero-shot text-to-speech (TTS) model. Practitioners fine-tune it to
adapt it to new speakers and languages, mostly with community recipes whose claims have never been checked. We
audit the released training code at a pinned commit of its maintained fork, and find six defects that silently
change what is trained:

- the DVAE codebook is overwritten by a running average in training mode, so any DVAE training changes what the
  codes mean;
- the voice-prompt region is masked from the GPT loss for only one sample per batch, and in the wrong time units;
- the released DVAE checkpoint loads only non-strictly;
- the tokenizer refuses languages outside its list;
- Hugging Face Whisper word timestamps cannot locate pauses;
- vocabulary extension initialises the newest token as a copy of an unrelated language tag.

The four defects in the training code are also present in the original project's final release. The other two
are properties of the released checkpoint and of Hugging Face Whisper. A seventh, introduced by the maintained
fork, silently rewrites the released checkpoint file the first time it is loaded, which breaks hash-based
provenance of every later measurement.

We release a three-stage recipe (DVAE, GPT, HiFi-GAN decoder) that repairs these defects. Each repair is guarded
by a test that fails when the repair is removed. Each recipe decision is graded by the strength of its evidence.
We also release an evaluation protocol: a frozen 180-sentence, five-language test set, paired A/B comparison
with bootstrap intervals, pre-registered gates, and target-speaker reference clips that training is barred from
using.

On the stock model, the English baseline is WER 0.060, CER 0.020, SECS 0.575 and UTMOS 4.04 across five reference speakers. On one real recording, a cut misplaced by ±1 s lowers the worst forced-alignment word score from ≥ 0.54 to ≤ 0.02, so such a cut is detectable. In smoke-scale runs, DVAE fine-tuning changes 81% of the discrete codes for identical audio, and freezing the codebook does not reduce this, demonstrating that DVAE fine-tuning invalidates the pretrained GPT. In target speaker adaptation with 6.4 minutes of aligned speech, Stage 2 GPT fine-tuning yields a statistically significant speaker similarity gain of +0.033 SECS (0.555 → 0.588, 95% CI [+0.019, +0.048], n=51) on held-out reference audio, passing all five pre-registered target-speaker gates. The same model fails its pre-registered preservation gate on five unseen voices: their similarity falls by 0.024 SECS (95% CI [−0.029, −0.018], margin 0.02), so the adaptation costs zero-shot cloning of other speakers.

**Index terms:** text-to-speech, voice cloning, fine-tuning, reproducibility, vector quantization, software
audit, evaluation methodology.

---

## 1 Introduction

Zero-shot multilingual TTS models clone an unseen voice from a few seconds of reference audio. XTTS [1] is a
widely used open model of this kind. Its released v2 checkpoint covers 17 languages, and its training code was
published in the Coqui TTS library. When zero-shot cloning is not good enough, for example for a particular
speaker, an accent or a language the model does not cover, practitioners fine-tune it. The available guidance
consists of community repositories, issue threads and forum posts. These sources disagree on basic questions:
whether the DVAE should ever be trained, whether short utterances should be added, whether the decoder should be
fine-tuned, and how to add a language. They rarely report controlled measurements. The upstream project stopped
active development in 2024 and has no maintainer to settle the questions.

A fine-tune is judged by listening, or by a loss curve. Neither can show a defect that silently changes the
training target. For example, a loss term that is masked for the wrong samples still trains, and a codebook that
drifts during training still reconstructs audio well. Such defects persist because nothing in the usual workflow
can detect them.

This work treats the fine-tuning recipe as a claim to be tested rather than a procedure to be followed. Our
contributions are:

- **C1, a code audit.** We find seven defects (F1–F7): four in the XTTS training code, one in its released
  checkpoint, one in the Whisper timestamps its data tools rely on, and one in the fork's checkpoint loader. Each
  is demonstrated by a test against the pinned upstream code. The four in the training code are also present in
  the original project's final release; the loader defect was introduced by the fork (Section 5.1).
- **C2, a repaired three-stage recipe.** It covers the DVAE, the GPT and the HiFi-GAN decoder. Each decision is
  tagged with an evidence level and guarded by a test (Section 3).
- **C3, a verifiable data pipeline.** Transcripts come from CTC forced alignment, cuts are planned at pauses, and a
  validator re-aligns every clip. This catches mis-cut and mis-transcribed clips, and we validate it with real
  audio (Section 5.4).
- **C4, an evaluation protocol.** It includes a frozen multilingual test set, metric definitions corrected for
  measurement artefacts we found in the stock model, paired A/B comparison with pre-registered gates, and
  hash-locked target-speaker hold-out clips (Sections 3.5, 5.2–5.3).
- **C5, a reproducibility discipline.** Every artifact and data source is hash-pinned, runs are deterministic per
  seed within an environment (across environments, 94.7% of takes are identical and means agree within intervals), and every bug fix follows a written protocol: the fix must fail when deliberately broken (Section 3.6).

The repository that accompanies this paper is the evidence for every claim in it. Appendix A maps each claim to
the file or test that supports it.

## 2 Background and related work

**XTTS architecture.** XTTS [1] builds on Tortoise [2] and has three trained components:

1. **A discrete VAE (DVAE).** It turns mel spectrograms into discrete codes, at 21.53 Hz with a 1024-entry
   codebook. The codebook was trained with 8192 entries and pruned to the 1024 most used [1].
2. **A GPT-2-style decoder-only transformer.** Given text tokens and a speaker conditioning, it predicts the DVAE
   codes autoregressively. The conditioning is the reference audio's mel spectrogram, compressed by a perceiver
   resampler into 32 latent vectors.
3. **A HiFi-GAN [3] decoder.** It synthesises 24 kHz audio directly from the GPT's final hidden states, conditioned
   on a speaker embedding. The DVAE's own decoder is therefore not used at inference.

The XTTS paper reports that about 10 minutes of fine-tuning data per speaker raised cross-lingual speaker
similarity from 0.585 to 0.717 [1]. It does not report the hyperparameters, how training references were chosen,
the decoder's full loss, or any procedure for adding languages.

**Vector-quantized representations.** VQ-VAE codebooks [4] are often updated by an exponential moving average
rather than by gradient. They are prone to collapse, where few codes stay in use [5]. When a downstream model is
trained on the codes, the code assignments become an interface. Any change to the quantizer or the encoder can
silently break that interface. Our finding F1 and our drift measurement (Section 5.5) concern this coupling.

**Adapting a pretrained model to a new language.** New-language adaptation must extend the token vocabulary.
Initialising new embedding rows from existing rows works better than random initialisation in language models
[6, 7]. Fine-tuning a multilingual TTS model in sequence on new languages causes forgetting of the earlier ones,
which is reduced by replaying data from those languages [8].

**Speaker adaptation and the decoder.** YourTTS [9] introduced a speaker consistency loss (SCL) for the vocoder.
Its erratum states that the loss's gradient did not reach the intended component, so the evidence for SCL is
weaker than its wide adoption suggests.

**Data preparation.** CTC forced alignment [10] with multilingual acoustic models (MMS [11]) gives word timings
from a known transcript. Whisper [12] is commonly used both for transcription and for automatic intelligibility
metrics.

**Evaluation of zero-shot TTS.** Following [1], we measure intelligibility with Whisper large-v3 (WER/CER),
speaker similarity as the cosine similarity of ECAPA2 [13] embeddings (SECS), and predicted naturalness with
UTMOS [14]. We compare models with paired bootstrap intervals [15].

**Reproducibility.** Reproducibility checklists have been proposed for machine-learning research [16]. Our
approach is closer to software verification: each claim is linked to an executable check, and the check is
itself shown to fail when its subject is broken.

## 3 Method

### 3.1 Evidence grading

No recipe decision is accepted because of who proposed it. Each decision carries one of five evidence levels:

| Level | Meaning | Treatment |
|---|---|---|
| E1 | Verified in the pinned upstream code, or measured in this repository | Build on it; guard it with a test |
| E2 | Peer-reviewed result (noted when single-study or out-of-domain) | Strong default; confirm on our setup |
| E3 | Preprint or adjacent domain | Reasonable default; confirm by experiment |
| E4 | Practitioner report: uncontrolled | Hypothesis with some support; experiment required |
| E5 | Our own hypothesis, untested | Test before it shapes a default |

A decision rises in level only through a test or a measured experiment. The full register of decisions and
their levels is maintained in `docs/recipe_design.md` (§3–§4). It also assesses eleven community claims. Two are
wrong, and most of the rest are unproven or right for a reason other than the one given.

### 3.2 Code audit

The upstream code is the maintained fork `idiap/coqui-ai-TTS`, pinned at commit `03339e5b`. Every finding was
also checked against the original `coqui-ai/TTS` at release v0.22.0 (tag commit `fa28f99f`), using recorded
commands and file hashes. For each finding we wrote two tests:

1. a test that **demonstrates** the upstream behaviour against the pinned code;
2. a **guard** that fails if our repair is removed.

The seven findings are summarised in Table 1 and their consequences are discussed in Section 5.1.

**Table 1.** Defects found in the upstream XTTS training code (F1, F2, F4, F6), its released DVAE checkpoint (F3), the Hugging Face Whisper timestamps used for data preparation (F5), and the maintained fork's checkpoint loader (F7).

| ID | Defect | Consequence | Repair |
|---|---|---|---|
| F1 | In train mode the DVAE quantizer overwrites its codebook on every forward pass with a running average (decay 0.99). The codebook is not learned by gradient. | Any DVAE training changes what the codes mean, whatever the learning rate. A pretrained GPT no longer matches the codes. | DVAE training is off by default. A trained DVAE is marked as requiring a GPT retrain, and Stage 2 and latent extraction refuse mismatched pairs. |
| F2 | The prompt-region loss mask in `gpt.py` reuses the loop variable `idx` from padding computation, masking only the final batch sample; additionally, prompt bounds are divided by 256 (mel frames) instead of 1024 (code stride). | For batch size $N > 1$, samples $0 \dots N-2$ compute autoregressive cross-entropy loss against prompt tokens already present in the conditioning prefix, while the final sample $N-1$ has incorrect prompt codes masked due to the 256 vs. 1024 coordinate mismatch. | Recompute mel loss from logits, masking prompt spans across all batch samples individually using `cond_idxs` in code coordinates (stride 1024). |
| F3 | The released `dvae.pth` lacks 3 of the model's 56 tensors (running-statistics buffers). Upstream loads it with `strict=False`. | Non-strict loading hides this mismatch and would hide any real one. | Strict loading with an exact allowlist of those three buffers. |
| F4 | The tokenizer raises an error for any unlisted language, and inference indexes a per-language character limit. | A model trained on a new language cannot run through stock inference. | A tokenizer subclass with registered languages, used in every code path. |
| F5 | Hugging Face Whisper word timestamps are contiguous. Separately, when a transcript omits words, CTC alignment stretches a neighbouring word over the extra speech. | Whisper timestamps cannot locate pauses, and missing text goes unnoticed. | Word times come from CTC forced alignment. Missing text is detected as frames the model labels non-blank but the alignment path assigns to blank. |
| F6 | Vocabulary extension gives new rows unit-variance random values, and copies the old last row (assumed to be `[START]`) onto the new last row. In XTTS-v2 the last row is the `[hi]` language tag. | The newest token, typically the new language's tag, starts as a copy of the Hindi tag. | New rows start at the mean of existing rows plus small noise. |
| F7 | The fork's `Xtts._load_checkpoint` (`TTS/tts/models/xtts.py:639-652`) converts a pickled `XttsConfig` to a dict and `torch.save`s the result over the original `model.pth`. The released file stores its config that way, so the first load rewrites it. | The released weights file changes on disk (SHA-256 `c7ea2000…` → `3077a05b…` in our runs 1–2), so later measurements can no longer be tied to the locked artifact by hash. | The conversion runs in memory; every loader applies the guard, and a static test fails if one doesn't. |

The audit also found three other upstream behaviours that the Stage 2 wrapper changes:

- the dataset silently substitutes sample 1 for any sample that fails to load;
- default model arguments do not match the released architecture;
- the trainer's `start_with_eval` option turns the first epoch into an evaluation, so N epochs train only N−1.

### 3.3 The three-stage recipe

**Stage 1: DVAE, off by default.** Because of F1, DVAE training is recommended only when a diagnostic shows the
stock DVAE fits the target audio clearly worse than reference speech. Any trained DVAE then requires a full
Stage 2 retrain. Training fails if the codebook collapses on validation data, consistent with dead-code dynamics in vector-quantized representations [4, 5]. A frozen-codebook mode, with the
quantizer fixed and the encoder trainable, is provided as a candidate mitigation; Section 5.5 tests it.

We define code drift as the fraction of validation code positions whose code changes:

$$\delta = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\left[c_i^{\text{ft}} \neq c_i^{\text{stock}}\right],$$

where $c_i^{\text{ft}}$ and $c_i^{\text{stock}}$ are the codes the fine-tuned and stock DVAE assign to position
$i$ of the same validation audio. Drift is logged at every validation step. We also log active codes, code
perplexity (the exponential of the code-usage entropy) and reconstruction loss [5].

**Stage 2: GPT.** Stage 2 is a thin wrapper over the upstream trainer, so the training loop is not reimplemented.
It adds four things:

- the per-sample prompt masking fix (F2) to prevent conditioning-to-target autoregressive leakage [1, 2];
- strict loading of the base weights;
- mean-initialised new text rows (F6) rather than random unit-variance rows, anchoring new tokens inside the pre-existing semantic embedding subspace [6, 7, 20];
- a registered-language tokenizer (F4); new-language text uses per-character tokens by default, with BPE merges or romanization as candidate alternatives [22].

New-language runs mix in replay data from the base languages to prevent catastrophic forgetting in lifelong multilingual TTS [8], and the language shares are reported. The
starting hyperparameters follow the upstream recipe: AdamW (β = 0.9, 0.96), weight decay 10⁻² on weights only,
learning rate 5 × 10⁻⁶, and a multi-step schedule. We treat these as defaults to be tuned against evaluation
loss, not as established values.

**Stage 3: HiFi-GAN decoder, optional.** The decoder's inputs are latents extracted from the GPT the decoder will
run with [1, 3]. The GPT is in eval mode and teacher-forced with the true transcript. Latents are interpolated over the
whole clip before segments are cut, so segments align exactly with the decoder's own interpolation. No upstream
discriminator weights exist, so the discriminators start from scratch [3]. They therefore train during a warm-up in
which the generator learns from mel loss only. A run that would never leave warm-up is refused. The best
checkpoint is chosen by validation mel loss. SCL [9] and a full-band (12 kHz) mel loss are available as options, off
by default. A Stage 3 result is kept only if it beats the stock decoder in the A/B comparison.

### 3.4 Data preparation and validation

Each clip's text must be exactly the words aligned to it. Transcripts come from corpus metadata or a human
transcript, aligned with MMS forced alignment [11]. Without a transcript, Whisper [12] phrases are used only to
bound the alignment windows.

**Cut planning.** Cuts are made only at pauses of at least 150 ms. Among plans whose segments all lie within
$[s_{\min}, s_{\max}]$, the planner maximises a lexicographic objective:

$$\text{score}(P) = \Big(\operatorname{round}\big(S_{\text{kept}}(P), 10^{-6}\big),\; \sum_{g \in \text{cuts}(P)} \min(g, g_{\text{cap}})\Big),$$

where $S_{\text{kept}}$ is the speech duration kept and $g$ is the pause length at each cut, capped at
$g_{\text{cap}} = 1$ s. Kept speech always dominates, and longer pauses only break ties. An earlier, weighted
objective let the pause preference outweigh speech and dropped words on real audio (Section 5.4). Regression
tests reproduce all three planner versions.

**Validation.** Every dataset passes a validator before every stage. It checks the schema, durations, text-token
limits, overlap with the evaluation set and reference voices, and target-speaker hold-out overlap. With
re-alignment enabled, the validator re-aligns each clip and rejects it if:

- any word scores below 0.05;
- more than three words score below 0.2 (calibrated on eight baseline recordings; Section 3.6);
- more than 0.1 s of speech is left unexplained by the transcript;
- a word is implausibly stretched.

### 3.5 Evaluation protocol

**Test set.** The frozen evaluation set (v1.2.0, SHA-256 `86c0183f…`) has 180 sentences:

- 60 English sentences, 10 in each of six categories: Harvard sentences, sibilant-heavy text, numbers, long
  complex sentences, prosody and questions, and short (1–3-word) utterances;
- 30 sentences each in Spanish, French, German and Hindi.

Five CMU ARCTIC speakers [17] serve as reference voices. Their reference clips are assembled from whole
utterances and pinned by hash. The licence file of each speaker is pinned too. The set is never regenerated
silently: a missing file is an error. The loader also recomputes the set's hash, so an edited set can't carry
its old hash into a manifest.

**Metrics (version 2).**

- **WER and CER:** from Whisper large-v3 with the standard text normaliser.
- **SECS:** the cosine similarity of ECAPA2 embeddings. ECAPA2 is independent of the model's own speaker encoder.
- **UTMOS:** reported only relative to the baseline.
- **Duration ratio:** measured against an estimate from the words XTTS is actually asked to say, after upstream's
  text cleaner. A clip is flagged too long only if it exceeds 2.2× the estimate *and* the estimate plus 1 s.
- **Headline SECS:** counts only clips of at least 2 s.

The last two definitions correct measurement artefacts found in the stock baseline (Section 5.3).

**Paired A/B comparison.** A fine-tuned model is compared with stock XTTS-v2 on identical inputs. Utterances are
paired by language, voice, sentence and seed. For each metric we compute the per-pair difference
$d = m_{\text{tuned}} - m_{\text{base}}$, its mean, and a 95% bootstrap interval from 10,000 resamples. The same
is computed per category and per voice.

**Pre-registered gates.** The gates were fixed before any fine-tuned model was measured:

- the tuned model may be no worse than stock by more than 0.02 WER, 0.01 CER, 0.1 UTMOS and 0.02 SECS on unseen
  voices (upper end of the oriented interval ≤ margin);
- it may produce no more failures than stock.

The margins were sized against the seed noise in the baseline, whose 95% interval half-widths were about 0.02 for
WER and SECS and 0.07 for UTMOS.

**Target-speaker evaluation.** For voice adaptation, the question is whether the fine-tune sounds more like the
target speaker than stock cloning does. Whole clips of each target speaker are held out (seeded) and assembled
into a 6–10 s reference clip. The SHA-256 of every held-out clip goes into a lock at a single canonical path. The
dataset validator refuses any training clip whose audio matches the lock, and evaluation reads target clips only
from that path, so the clips it scores are the ones training refuses. Target-speaker and eval-voice measurements
are separate runs, so the forgetting gates are never diluted by the target speaker. The target gate requires the lower end
of the SECS interval to be above zero.

### 3.6 Reproducibility discipline

- **Pinned inputs.** All six upstream model artifacts are pinned by hash in `artifacts.lock.json`. Every data
  file must have a SHA-256 pin before it can be downloaded: an unpinned file raises an error. GitHub sources use
  commit-pinned URLs. For the archive.org source (a LibriVox recording, public domain), each SHA-256 was taken
  from a download whose SHA-1 matched the value archive.org publishes, which carries that provenance over
  without relying on SHA-1 for the check.
- **Determinism.** Evaluation manifests record the eval-set hash, voice locks, seeds, languages, ASR model and a
  code fingerprint. Comparisons refuse manifests whose inputs differ.
- **Bug-class protocol.** Every change follows eight steps:
  1. reproduce the bug with a failing test;
  2. name its class;
  3. search the code for siblings of that class;
  4. fix it at the single source of truth;
  5. take external facts from recorded command output;
  6. show the fix's test fails when the fix is deliberately broken (a mutation check);
  7. measure model behaviour before and after;
  8. link the finding to its guarding test.
- **Thresholds.** Thresholds come from a baseline, a publication or a stated requirement, never from the current
  output. One exception is disclosed: the alignment-rejection thresholds (Section 3.4) were calibrated on the eight
  baseline recordings (`[E5]`) that Section 5.4 analyzes. The floor was set to 0.05 on JFK cut shifts, but the
  weak-word allowance was increased from one to three after observing that clean, accented non-native speech
  (`arctic_ksp`, Indian English) produced three weak function words scoring down to 0.062 (a narrow 0.012 margin above
  the floor). The unexplained-speech limit was retained at 0.10 s, where all clean recordings measure 0.00 s. These
  values are therefore in-sample calibrations on those recordings, not independent validations of them. A
  pre-registered held-out run then validated them out-of-sample for clean English speech only (Section 5.4,
  Table 3c).
- **Continuous integration.** CI runs linting and the full test suite. Tests that need real weights, GPU or
  licensed audio run in a Colab notebook, and their outputs are archived as evidence.

## 4 Experimental setup

**Hardware and software.**

- **Baselines (runs 1–2):** Tesla T4 GPU; Python 3.13.16, PyTorch 2.11.0 (CUDA 13.0), transformers 5.20.2,
  `coqui-tts` at the pinned commit.
- **Data preparation and training (runs 3–4):** a single Colab GPU.

The full environments (`pip freeze`) are archived with the evidence.

**Model.** The released XTTS-v2 checkpoint, with every artifact verified against the hash lock.

**Data for the smoke runs.** Eight clips:

- six LJSpeech clips [18] (public domain), with the corpus transcripts;
- two clips cut from an 11 s recording of J. F. Kennedy's inaugural address (a US federal government work,
  public domain), with a human transcript and MMS forced alignment.

The smoke runs test whether each stage runs correctly. They are not meant to improve quality.

**Run settings.**

| Stage | Settings |
|---|---|
| Stage 1 | 20 epochs, batch 4; normal and frozen-codebook modes from the released DVAE [1, 5] |
| Stage 2 | 3 epochs of 3 optimizer steps (9 steps); 6 train and 2 evaluation clips; batch 2, no accumulation; fp32 |
| Stage 3 | 60 epochs, batch 4; discriminator warm-up 40 steps; mixed precision on CUDA [3] |

**Baseline evaluation.** Full multilingual baseline: 5 languages (`en`, `es`, `fr`, `de`, `hi`), 12 sentences × 5 ARCTIC voices [17] × 3 seeds = 900 utterances, scored under metrics version 2 on locked checkpoint `c7ea2000…` (expanded from the initial 120-utterance English pilot). Intelligibility is measured using Whisper large-v3 WER/CER with standard normalisation [12], speaker similarity via ECAPA2 cosine similarity [13, 21], and naturalness via UTMOS [14], following the evaluation protocol established in [1]. Paired bootstrap intervals follow Efron and Tibshirani [15].

## 5 Results to date

### 5.1 The training-code defects date from the original release

All seven findings were reproduced by tests against the pinned fork. F1, F2, F4 and F6 are also present at the
corresponding lines of the original `coqui-ai/TTS` v0.22.0. The silent sample substitution, the non-strict loads
and the half-clip prompt for short clips are present there too. F3 is a property of the released checkpoint
itself, and F5 of Hugging Face Whisper, so neither depends on the code base. The four training-code defects
therefore date from Coqui's own release and are inherited by any recipe built on it. F7 is the exception: v0.22.0's
checkpoint loader has no such rewrite, so the fork introduced it. It explains the one provenance gap in our
evidence: the baseline manifests of Colab runs 1–2 record `model.pth` as `3077a05b…` rather than the locked
`c7ea2000…`, although those runs' own artifact checks found the locked file. The rewrite replaces the pickled
config object with a dict; whether the tensors are byte-identical has not been checked, so all reported baseline
numbers come from Run 5, which measured the locked file with the guard in place. Each repair has a guard test that fails when the
repair is removed.

### 5.2 Artifact integrity and determinism

All six upstream artifacts matched the hash lock in two independent runs. The mel normalisation statistics were
identical (maximum absolute difference 0.0) across three sources: the packaged file, the Hugging Face file and the
tensor stored inside `model.pth`. Two runs with the same seeds agreed exactly on every metric, with identical ASR
transcripts and durations. Different seeds changed the audio: 47 of 50 durations differed between seeds 42 and
123. Within one environment the pipeline is therefore deterministic per seed, and seed variance is real.

Across environments it is not bit-identical. The Phase 8 rerun of the 5-language baseline (a later commit and a new
GPU session; `docs/evidence/phase8_reproducibility/`) produced identical audio for 852 of 900 takes (94.7%); the
other 48 differ. Every language's means agree with the published run within the rerun's 95% tolerance, but one
case shows how little that can say: one Spanish take ran away in the rerun (audio 8.96 s → 17.17 s; Whisper's transcript repeats the sentence's
opening 25 times, 262 words for a 12-word sentence, WER 20.8). It accounts for the whole net change in Spanish mean
WER, 0.099 to 0.213 (four other takes changed slightly, in both directions), while the median stays 0.0; without that take the two runs
match (WER 0.100 vs 0.098, CER 0.083 vs 0.082).
We therefore claim determinism per seed within an environment, and agreement within intervals across environments,
not bit-identical reproduction across environments.

### 5.3 Stock XTTS-v2 baseline and measurement artefacts

**Table 2.** Stock XTTS-v2, full multilingual baseline across 5 languages (900 utterances: 12 sentences × 5 ARCTIC voices × 3 seeds, measured on locked checkpoint `c7ea2000…`; metrics version 2). Values are mean ± 95% confidence interval half-width ($1.96 \cdot \text{std}/\sqrt{n}$). Headline SECS counts clips ≥ 2 s (English $n = 150$, all other languages $n = 180$).

| Language | WER (Whisper large-v3) | CER | SECS (≥ 2 s) | SECS (all) | UTMOS | Duration ratio |
|---|---|---|---|---|---|---|
| **English (`en`)** | 0.060 ± 0.018 | 0.020 ± 0.009 | 0.575 ± 0.017 | 0.537 ± 0.019 | 4.04 ± 0.06 | 1.06 ± 0.05 |
| **Spanish (`es`)** | 0.099 ± 0.075 | 0.083 ± 0.049 | 0.511 ± 0.006 | 0.511 ± 0.006 | 3.29 ± 0.06 | 1.08 ± 0.07 |
| **French (`fr`)** | 0.043 ± 0.012 | 0.031 ± 0.009 | 0.450 ± 0.007 | 0.450 ± 0.007 | 3.30 ± 0.06 | 0.94 ± 0.05 |
| **German (`de`)** | 0.053 ± 0.017 | 0.032 ± 0.011 | 0.512 ± 0.009 | 0.512 ± 0.009 | 3.44 ± 0.04 | 1.06 ± 0.03 |
| **Hindi (`hi`)** | 0.101 ± 0.011 | 0.047 ± 0.007 | 0.542 ± 0.011 | 0.542 ± 0.011 | 3.45 ± 0.04 | 1.29 ± 0.02 |
| **Multilingual Mean** | **0.071** | **0.043** | **0.518** | **0.510** | **3.50** | **1.09** |

The multilingual results in Table 2 establish clear language-dependent baselines: English achieves the highest quality (UTMOS 4.04, WER 0.060), while Hindi has a higher typical word error rate (0.101, median 0.086) across all of its takes. Spanish's mean
WER (0.099) is not a typical-case difference: it comes from 4 runaway takes (WER above 1.0, up to 6.5); without
them Spanish is at 0.043, the same as French, and its median is 0.0. Mean WER and CER are unbounded per take, so
for every comparison in this paper we also report medians or runaway counts where single takes dominate. Across the 900 utterances, the pipeline had zero silent dropouts (`is_empty`) and zero `NaN` values, with 7 runaway-duration flags (`is_too_long`: 5 Spanish, 2 French).

The evaluation protocol and Metric Version 2 rules were originally established during the initial 120-utterance English category pilot (Colab Run 2; 60 sentence/voice pairs × 2 seeds; `docs/evidence/colab_run2/README.md`), where the model made no recognised errors on four categories and all sibilant WER stemmed from a single tongue-twister sentence that challenged ASR. That pilot highlighted three measurement artefacts that required methodological corrections:

1. **Duration estimated from written digits.** "4,800,000" is 9 characters but about 2 s of speech. The numbers
   category therefore scored a duration ratio of 1.61. Estimating duration from the cleaned words lowers the
   ratio to 0.82.
2. **One-word utterances flagged too long.** A 0.5 s floor on the duration estimate flagged the normal one-word
   answer "Yes." (1.15 s) as too long. Requiring 1 s of absolute excess removes the false flag.
3. **SECS on very short clips.** SECS on 0.7–1.15 s clips is 0.33, against 0.48–0.62 for every other category.
   ECAPA2 does not get enough speech from such clips for a reliable embedding. Short-text quality must therefore
   be judged with WER, CER, absolute duration and UTMOS, not SECS.

In that pilot, every short clip lasted 0.69–1.15 s, and every other clip at least 2 s, establishing the 2 s cutoff to separate categories cleanly. Per voice, SECS (all clips) ranged from 0.44 to 0.61, confirming that voice must be a pairing factor in any comparison.

### 5.4 Forced-alignment validation detects misplaced cuts

**Planner regression.** An earlier version of the cut planner scored the audio it kept. Cutting at a long pause
therefore looked worse, because the pause itself is dropped. On the 11 s JFK recording, that planner cut at a
tight word boundary and lost a clip. Its replacement, a weighted pause bonus, in turn dropped the opening words
"And so,". The lexicographic objective (Section 3.4) gives two clips covering 89% of the recording. The resulting
eight-clip sample set was prepared and validated with no manual edits.

**Shifted-cut test.** We prepared JFK with MMS forced alignment and a 2.5 s minimum clip length, which gives
three clips (the sample set above uses 3.0 s and gives two). We moved the end of the first clip by ±1 s and
re-aligned every clip as the validator does (Table 3). Only the first clip was shifted.

**Table 3.** Re-alignment word scores for the first JFK clip, "And so, my fellow Americans: ask".

| Clip end | Lowest word score | Words < 0.2 | Unexplained speech | Validator |
|---|---|---|---|---|
| As cut | 0.866 | 0 | 0.00 s | accepts |
| +1 s | 0.012 | 5 | 0.04 s | rejects |
| −1 s | 0.018 | 5 | 0.00 s | rejects |

Across all three unshifted clips, the lowest re-aligned word score was 0.54. The word "for" scored 0.12 when the
whole 11 s recording was aligned in one pass, but 0.66 when re-aligned within its own clip. A low score from a
long alignment window is therefore not, by itself, evidence of a bad clip. For clean native recordings (JFK,
LJSpeech, and standard ARCTIC voices), clean clips maintain a wide score margin ($\ge 0.54$). However, on accented
non-native speech (`arctic_ksp`, Indian English), unstressed function words score down to $0.062$ (only $0.012$
above the $0.05$ floor), registering three weak words below $0.20$. Accommodating this speaker without dropping the
0.05 safety floor required calibrating the weak-word allowance to three (`max_weak_words = 3`), narrowing the
detection margin and trading off sensitivity to short tail truncations. Retaining the unexplained-speech limit at
0.10 s ensures that clean recordings (all measuring 0.00 s) pass while trailing speech intrusions above 100 ms are
flagged. These thresholds are therefore in-sample calibrations (`[E5]`), not an independent validation of them;
the held-out run below supplies that validation for clean English speech.

**Multi-speaker and cross-accent calibration across 8 baseline voices.** To characterize these thresholds across a wider range of acoustic conditions in-sample, we evaluated an 8-speaker calibration set on an NVIDIA A100 GPU spanning two languages (English and Castilian Spanish) and five accent profiles (`jfk_english_male`, `arctic_slt_english_female`, `arctic_ksp_english_indian_accent`, `arctic_bdl_english_male`, `arctic_clb_english_female`, `arctic_awb_english_scottish`, `ljspeech_female`, and `spanish_cuentos_male`).

On this calibration set, all eight clean recordings passed (accented speech in `arctic_ksp` passed with 3 weak words, minimum score 0.062, within the calibrated allowance of at most 3 words below 0.20; all clean clips measured 0.00 s unexplained speech). Across an evaluation matrix of 88 controlled corruption tests (11 perturbations per speaker across edge truncations, unspoken transcript word insertions, and trailing speech spillover), the benchmark characterized empirical detection performance and identified specific failure modes (Table 3b; detailed in `docs/evidence/phase6/`):

**Table 3b.** Observed rejection rates across 88 controlled corruptions on the 8 baseline calibration speakers (QualityConfig: min_word_score 0.05, weak_word_score 0.20, max_weak_words 3, max_unexplained_speech_s 0.10 s).

| Corruption Category | Mutation Variant & Severity | Observed Rejection Rate | Failure Mode & Mechanism |
|---|---|:---:|---|
| Edge Truncation | 100 ms tail cut | 3/8 (37.5%) | 5 passed: sub-250 ms cuts clip trailing aspiration without dropping vowel nucleus below 0.05 |
| | 250 ms tail cut | 5/8 (62.5%) | 3 passed: partial coda loss produces word scores in 0.08–0.16 weak tolerance band |
| | 500 ms tail cut | 2/8 (25.0%) | 6 passed: non-monotonic dip; deeper cut removes terminal word and allows trellis realignment across earlier speech without triggering < 0.05 floor |
| | 1,000 ms tail cut | 7/8 (87.5%) | 1 passed: final word eliminated, prior word boundary shifts |
| | 500 ms head cut | 7/8 (87.5%) | 1 passed: initial consonant cluster partially preserved |
| | 1,000 ms head cut | 8/8 (100.0%) | 0 passed: opening word onset completely lost; score drops to 0.00–0.04 (< 0.05 floor) |
| Transcript Insertion | Head function word | 7/8 (87.5%) | 1 passed: weak-word tolerance absorption (score 0.125 on LJSpeech) |
| | Mid content word | 6/8 (75.0%) | 2 passed: phonetic substring match (/dʒər/, /tr/ in "extraordinary" scores 0.234–0.237) |
| | Tail content word | 4/8 (50.0%) | 4 passed: trailing silence emissions average 0.06–0.16, absorbed by $\le 3$ weak-word limit |
| Speech Spillover | Sub-threshold (+100 ms) | 0/8 rejected (0%) | 8/8 correctly accepted: sub-100 ms tolerance functioning as designed |
| | Cross-threshold (+500 ms) | 1/8 rejected (12.5%) | 1 rejected (Spanish: 140 ms unexplained speech > 0.10 s floor); 7 passed (active speech 80–85 ms, remainder pause) |

Tail truncation exhibits a non-monotonic detection pattern (37.5% at 100 ms, 62.5% at 250 ms, 25.0% at 500 ms, and 87.5% at 1,000 ms): while 250 ms cuts directly clip terminal word nuclei dropping scores to 0.01–0.04 (< 0.05 floor), a 500 ms cut often eliminates the terminal word entirely or lands in an inter-word pause, enabling the CTC trellis to realign earlier tokens across remaining frames without triggering the 0.05 floor (the two rejected 500 ms cases scored 0.01 on their truncated terminal words), before 1,000 ms cuts reliably truncate full words and shift prior boundaries.

A frame-level diagnostic probe on GPU confirmed that trailing speech frames are assigned to CTC blanks rather than stretching the terminal word. Under the 0.10 s unexplained-speech limit, the Spanish spillover case (measuring 140 ms of active speech) is detected and rejected, while the remaining seven cases had active trailing speech lasting only 80–85 ms (the rest consisting of inter-utterance pause), which fell below the 0.10 s limit.

**Held-out validation on unseen speakers.** The thresholds above were then frozen and measured out-of-sample
under a pre-registered protocol (`docs/evidence/phase6_heldout/README.md`, committed before the run), in the locked
environment (`torch 2.14.1+cu130`, CUDA 13.0), on speakers and text that played no part in setting them: CMU ARCTIC
`rms` and `jmk` (utterances `a0003`–`a0022`, transcripts from the corpus prompt file) and LibriVox openings in French
and German (transcripts checked by listening). In the first run, the listening checks were recorded in the working tree
and committed with the results, not before the run. Two reruns from fully committed trees reproduced every score and verdict
exactly, and verified all six source downloads against the pins, so the results below do not depend on that
deviation. The second rerun (`f4106c5`) also records the vocabulary, alignment-model weight and code hashes, checked
against the artifact lock and an independent download; it is the reference record.

**Table 3c.** Held-out results. A corruption counts as detected only at a rejection rate of at least 80%
(pre-registered); intervals are 95% Clopper–Pearson.

| Measure | Result | Pre-registered verdict |
|---|---|---|
| Clean acceptance, English | 59/60 (98.3%, [0.911, 1.000]); silence controls 17/17 | Validated |
| Clean acceptance, French / German | 4/4 ([0.398, 1.000]) / 3/3 ([0.292, 1.000]) | Too few clips to validate |
| Head speech cut 600 / 300 / 150 ms | 100% / 93.8% / 50.0% rejected | Detected / detected / limitation |
| Tail speech cut 600 / 300 / 150 / 50 ms | 93.3% / 43.8% / 12.5% / 0% rejected | Detected / limitation / limitation / limitation |
| Inserted word at the start / middle / end | 100% / 76.5% / 64.7% rejected | Detected / limitation / limitation |
| Next-utterance spillover 0.50 s / 0.05 s | 80.0% rejected ([0.519, 0.957]) / 14/15 passed | Detected at the bar / passes as intended |

The out-of-sample evidence (`[E1]`) therefore covers **clean acceptance on English**: at most about 9% of clean
English clips are wrongly rejected, at 95% confidence. The single rejection is `arctic_jmk` "Hardly were **our**
plans made public", where "our" scored 0.022 (below the 0.05 floor). French and German are reported but not
validated. Detection is reliable only for large corruptions: head cuts of 300 ms or more, the 600 ms tail cut, an
inserted opening word, and 0.5 s of spillover. Shorter tail cuts and inserted mid- or end-of-sentence words often
pass, so the validator is a safety net against gross mis-cuts, not a guarantee of exact boundaries. The recorded
scores show why (exploratory, `docs/evidence/phase6_heldout/README.md` §6.4): almost every rejection comes from the
0.05 floor, the weak-word allowance of three never triggers on held-out clean speech, and the corruptions that pass
typically have a single weak word.

### 5.5 Training smoke runs with real weights

**Table 4.** Smoke-scale runs on the eight-clip sample set.

| Stage | Measure | Start | Best | Note |
|---|---|---|---|---|
| 2 (GPT) | Eval loss (text CE + mel CE) | 3.200 | 2.972 (step 9, the last optimizer step) | Strict load of base weights; export completed |
| 3 (HiFi-GAN) | Validation mel L1 | 1.553 | 1.350 (step 106) | Rose to 1.367 by step 120 |
| 1 (DVAE, normal) | Val. reconstruction | 0.243 | 0.105 | 141 active codes (stock 136); perplexity 114.3 (114.6) |
| 1 (DVAE, frozen codebook) | Val. reconstruction | 0.243 | 0.116 | 147 active codes; perplexity 123.2 |

All three stages load the released weights strictly, train, validate and save a best-by-validation checkpoint.
In Stage 3, validation loss reached its minimum and then rose within 120 steps on this small set, so selecting
the best checkpoint rather than the last one is necessary. The first four Stage 3 measurements (steps 2–8) were
identical to full floating-point precision. Run 5 logged the cause (`docs/evidence/run5/hifigan_metrics.jsonl`):
PyTorch's fp16 gradient scaler starts at a loss scale of 65536, and every one of the first 8 generator steps
overflowed, so each was skipped and the scale halved, down to 256. Validation was flat for exactly those steps and
improved from step 10. This is the scaler's normal start-up behaviour, not a defect; it costs 8 steps, which
matters only in short smoke runs.

**DVAE fine-tuning changes most codes.** Neither DVAE mode collapsed the codebook, and both improved
reconstruction. In both modes, however, code drift reached $\delta = 0.815$: 81.5% of validation code positions
changed code for identical audio. Drift grew steadily with training; in normal mode it was 3.5% at step 2, 46% at
step 20 and 81% at step 40. Freezing the quantizer did not reduce drift, because the trainable encoder moves its
outputs to different codebook entries. On synthetic data the same mode drifted 47%.

A GPT pretrained on stock codes would see mostly different tokens. Two consequences follow, both now enforced by
the recipe: (i) DVAE training without a full GPT retrain is invalid, regardless of how good the reconstruction
looks; (ii) the frozen-codebook mode cannot be used to avoid that retrain. Encoder-frozen or decoder-only
variants remain untested.

### 5.6 Target speaker voice adaptation (Stage 2 GPT)

We evaluate target speaker adaptation on Moira Fogarty reading *The Art of War* (Chapters 1–2, 8.4 minutes total audio, public domain). The audio was segmented using CTC forced alignment with punctuation-aligned natural phrase grouping configured to 3–11 s bounds (measured clip durations 3.002–9.026 s, retaining pause disagreements), yielding 85 clips. Two reference clips (7.6 s total) were strictly sequestered into the canonical holdout directory `eval/target_voices/` and barred from training. The remaining 83 clips (6.4 min speech) formed the training set (`dataset_train.jsonl`, split into 75 train clips and 8 eval clips).

The Stage 2 autoregressive GPT was fine-tuned for 10 epochs on an NVIDIA A100 GPU using AdamW with per-sample prompt masking (F2 fix). Evaluation loss improved from 3.947 to 3.102 (−21.4%).

We evaluated the fine-tuned model against stock XTTS-v2 via paired A/B comparison on held-out reference audio across multiple seeds, evaluating against pre-registered quality gates (`eval/gates_target.json`) across $N = 60$ ($20 \text{ sentences} \times 3 \text{ seeds}$) designed to provide tight statistical bootstrap bounds (Table 5).

**Table 5.** Target speaker voice adaptation A/B evaluation (stock XTTS-v2 vs. Stage 2 fine-tuned model on held-out reference audio; paired bootstrap 95% confidence intervals, $N=60$ evaluations with SECS $n=51$).

| Metric | Pre-registered Gate | Stock Base ($N=60$) | Tuned Model ($N=60$) | Paired Difference $\Delta$ | 95% Bootstrap CI | Gate Status |
|---|---|---|---|---|---|---|
| **SECS** ($n=51$) | `better` (margin 0.0) | 0.5545 | **0.5875** | **+0.0329** | [+0.0191, +0.0480] | **PASS** |
| **WER** | `not_worse` (margin 0.02) | 0.0383 | **0.0375** | **−0.0008** | [−0.0096, +0.0079] | **PASS** |
| **CER** | `not_worse` (margin 0.01) | 0.0098 | 0.0105 | +0.0008 | [−0.0045, +0.0060] | **PASS** |
| **UTMOS** | `not_worse` (margin 0.10) | 4.2589 | 4.2149 | −0.0439 | [−0.0831, −0.0065] | **PASS** |
| **Duration Ratio** | Informational | 0.9876 | 0.9649 | −0.0227 | [−0.0528, +0.0073] | — |
| **Failures** | `not_worse` (margin 0) | 0 / 60 | 0 / 60 | 0 | [0.0, 0.0] | **PASS** |

On the target speaker, all five gates passed: SECS gained $+0.0329$ (95% CI $[+0.0191, +0.0480]$, so the
`better` rule holds), WER and CER stayed within their margins, and UTMOS fell by $0.0439$ (95% CI $[-0.0831,
-0.0065]$), a statistically significant decrease that stays inside its 0.10 margin. These target gates are the rule
set `eval/gates.json` specified before Run 6 (commit `603e0d3`): its four non-regression gates plus its
target-speaker template (`secs`, `better`, margin 0). `eval/gates_target.json`, committed with the results, applies
exactly that set.

The same fine-tuned model was also measured against stock XTTS-v2 on the five unseen CMU ARCTIC voices
(20 sentences × 5 voices × 3 seeds, $N=300$), against the pre-registered gates in `eval/gates.json` (Table 5b).
This is the forgetting check: adapting to one speaker must not degrade cloning of others.

**Table 5b.** Preservation on unseen voices (stock XTTS-v2 vs. the same Stage 2 fine-tuned model; paired
bootstrap 95% confidence intervals, $N=300$, SECS on the $n=253$ pairs where both takes are $\ge 2$ s).

| Metric | Pre-registered Gate | Stock Base | Tuned Model | Paired Difference $\Delta$ | 95% Bootstrap CI | Gate Status |
|---|---|---|---|---|---|---|
| **SECS** ($n=253$) | `not_worse` (margin 0.02) | 0.5743 | 0.5507 | −0.0236 | [−0.0290, −0.0181] | **FAIL** |
| **WER** | `not_worse` (margin 0.02) | 0.0697 | 0.0419 | −0.0278 | [−0.0530, −0.0067] | PASS |
| **CER** | `not_worse` (margin 0.01) | 0.1266 | 0.0452 | −0.0813 | [−0.1821, −0.0077] | PASS |
| **UTMOS** | `not_worse` (margin 0.10) | 4.0517 | 4.1467 | +0.0950 | [+0.0603, +0.1305] | PASS |
| **Failures** | `not_worse` (margin 0) | 6 / 300 | 3 / 300 | −3 | — | PASS |

The preservation comparison **fails** its pre-registered SECS gate (`passed: false` in
`docs/evidence/run6_target_speaker/ab_report_preservation.json`): similarity to unseen voices fell by $0.0236$,
beyond the 0.02 margin, and so is the interval's lower end (−0.0290), which is what the `not_worse` rule
tests; the upper end (−0.0181) is within the margin. Fine-tuning on 6.4 minutes of one speaker
measurably reduced zero-shot cloning of other speakers. We report this as a failed gate, not a near miss: the
margin was fixed in advance, and this is the regression it exists to catch.

The WER, CER and failure gates passed, but their mean differences should not be read as broad gains in
intelligibility. Mean WER and CER are unbounded per take and are dominated by a few runaway takes: on stock, the
mean CER of 0.1266 is strongly influenced by a few takes (CER 12, 10 and 5 contribute 0.090 of it), and both models have a median WER and CER of 0.0.
Excluding takes with WER above 1.0, mean WER is 0.057 (stock) and 0.042 (tuned). The fine-tuned model produced
fewer runaway takes (3 failures vs 6) on these voices; that is the substance of those three results.

In short: 6.4 minutes of aligned speech produced a statistically significant gain in similarity to the target
speaker, at a measurable cost in similarity to unseen speakers. A recipe change that keeps the target gain while
passing the preservation gate (for example, base-language replay, A7, or fewer adaptation steps) is future work.

## 6 Planned experiments (pre-registered)

Each experiment runs through the evaluation harness against stock XTTS-v2, with at least three seeds and
bootstrap intervals. Each decision rule below is fixed before the experiment runs.

**Status of the suite.** All eight ablations were pre-registered (`eval/pre_registered_ablations.json`, version 2,
committed before any of them ran). Only A3 has been run. For A1, A2 and A4–A8 the recipe uses defaults supported by
the literature cited in Table 6; those choices stay at evidence level E2/E3 (§3.1) until the ablation runs on this
model. A3 differs from the others because it tests a defect we found (F2), not a design choice; we know of no
earlier report of this defect or measurement of its effect.

**Table 6.** Pre-registered ablation study designs (A1–A8).

| ID | Question | Design | Primary metric | Decision rule | Status |
|---|---|---|---|---|---|
| A1 | What fixes short-text hallucination? | Short-clip share (0/5/15%) × minimum clip length (0.5/0.3 s) × minimum conditioning length (3/1 s) | CER on 1–3-word text; SECS on held-out speakers | Cheapest setting bringing short-text CER within the long-text interval without lowering SECS beyond the gate | **Not run**; default from Casanova et al. (*Interspeech* 2024) [1]; Chen et al. (2024) [19] |
| A2 | When is DVAE training worth it? | Stock vs trained vs frozen-codebook DVAE, each with a full Stage 2 retrain | Code usage, reconstruction, then end-to-end CER/SECS/UTMOS | Recommend Stage 1 only where end-to-end metrics improve beyond the interval | **Not run**; default from Zheng et al. (*ICCV* 2023) [5]; Casanova et al. (*Interspeech* 2024) [1] |
| A3 | Does fixing F2 matter? | Upstream vs per-sample prompt masking | Short-text CER, SECS, duration failures | Retain the fix when the pre-registered non-regression gates pass; report the effect either way | **Run (§6.1)**: non-regression gates pass on unseen voices; the fix is retained |
| A4 | Same-clip vs different-clip prompt | Masked same-clip vs same-speaker different-clip prompt | SECS (held-out speakers), CER | Adopt the better default | **Not run**; default from Casanova et al. (*ICML* 2022) [9] |
| A5 | New-script tokenisation and row initialisation | BPE vs per-character vs romanisation; mean vs random rows | CER; tokens per sentence; base-language regression | Best CER that passes the base-language check | **Not run**; default from Minixhofer et al. (*NAACL* 2022) [7]; Dobler et al. (*EMNLP* 2023) [6]; Ebing et al. (2026) [22] |
| A6 | Does punctuation–pause consistency matter? | With vs without commas inserted at long pauses | Pause placement, CER, UTMOS | Make the check blocking if the effect exceeds the interval | **Not run**; default from Saeki et al. (*Interspeech* 2022) [14] |
| A7 | Replay ratio for new languages | 0/10/25% base-language replay | Target CER; base-language CER/SECS | Smallest ratio with no base-language regression beyond the gate | **Not run**; default from Yang et al. (*ICASSP* 2022) [8] |
| A8 | Is Stage 3 worth it? | Stock vs fine-tuned decoder (± SCL, ± full-band loss) after a new-language GPT run | UTMOS, SECS, CER | Keep Stage 3 only if it beats stock beyond the interval | **Not run**; default from Kong et al. (*NeurIPS* 2020) [3]; Casanova et al. (*Interspeech* 2024) [1] |

### 6.1 Ablation A3: upstream vs per-sample prompt masking (F2)

We ran A3 on an NVIDIA A100-SXM4-40GB GPU with seeds `1, 2, 3`, comparing two Stage 2 fine-tunes that differ only
in prompt masking: **upstream masking**, which reproduces F2 (only the last sample of each batch is masked, in
conditioning-frame units), and **per-sample masking**, our fix (every sample's prompt masked in code units).
Both were evaluated with batched Whisper `openai/whisper-large-v3` (batch size 16) in two regimes:
1. **Target speaker** ($N=60$): held-out reference audio of Moira Fogarty, 20 prompts × 3 seeds (Table 7a).
2. **Unseen voices** ($N=300$): the 5 CMU ARCTIC voices, 20 prompts × 3 seeds (Table 7b). This is the regime the
   pre-registration names for A3's decision.

**Deviations from the pre-registration.** (i) It specifies "paired N=300 per seed, across 3 seeds" (900 pairs);
the run has 300 pairs in total (5 voices × 20 prompts × 3 seeds). (ii) It lists duration-ratio failures as a
primary metric; they are reported (Tables 7b, 7c) but have no gate of their own beyond the shared failures gate.
The target-speaker regime was not part of the pre-registration and is reported as an additional analysis.

**Table 7a.** Target speaker ($N=60$ paired takes on held-out audio; upstream masking is the A/B base, per-sample the candidate).

| Metric | Upstream Masking (F2) | Per-Sample Masking (fix) | Paired Difference $\Delta$ | 95% Bootstrap CI | Gate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Short-prompt CER ($n=9$) | 0.5556 | 0.0000 | −0.5556 | [−1.6667, 0.0000] | PASS |
| Overall CER | 0.0946 | 0.0083 | −0.0863 | [−0.2552, +0.0001] | PASS |
| Overall WER | 0.0529 | 0.0342 | −0.0187 | [−0.0550, +0.0018] | PASS |
| UTMOS | 4.2418 | 4.2134 | −0.0285 | [−0.0778, +0.0177] | PASS |
| Failures | 0 / 60 | 0 / 60 | 0 | — | PASS |
| SECS ($n=50$) | 0.5917 | 0.5748 | −0.0170 | [−0.0297, −0.0040] | **FAIL** |

**Table 7b.** Unseen voices ($N=300$ paired takes across 5 CMU ARCTIC voices).

| Metric | Upstream Masking (F2) | Per-Sample Masking (fix) | Paired Difference $\Delta$ | 95% Bootstrap CI | Gate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Short-prompt CER ($n=45$) | 6.3111 | 0.1741 | −6.1370 | [−16.7003, −0.2667] | PASS |
| Overall CER | 0.9601 | 0.0374 | −0.9228 | [−2.5863, −0.0368] | PASS |
| Overall WER | 0.8424 | 0.0422 | −0.8002 | [−2.3312, −0.0146] | PASS |
| UTMOS | 4.1150 | 4.1273 | +0.0123 | [−0.0187, +0.0441] | PASS |
| Failures (too long) | 7 / 300 | 3 / 300 | −4 | — | PASS |
| SECS ($n=255$) | 0.5690 | 0.5628 | −0.0062 | [−0.0110, −0.0014] | PASS |

**The means in Tables 7a–7b are dominated by single takes.** WER and CER are unbounded per take, and the median
is 0.0 for both variants in both regimes. On unseen voices, one upstream take on `arctic_ksp` (seed 1) has WER
221 (the model kept generating speech) and one on `arctic_bdl` (seed 2) has WER 11; together they contribute 0.77
of the 0.84 mean. In the target regime,
the 0.5556 short-prompt CER is one take with CER 5.0 out of 9. Table 7c reports statistics that a single take
cannot move.

**Table 7c.** A3 without the influence of single takes. "Runaway" means WER above 1.0. Fisher exact tests are
two-sided and exploratory: these comparisons were chosen after seeing the data and were not pre-registered.

| Regime | Statistic | Upstream (F2) | Per-sample (fix) | Fisher $p$ |
| :--- | :--- | :---: | :---: | :---: |
| Unseen voices | Median WER / CER | 0.000 / 0.000 | 0.000 / 0.000 | — |
| Unseen voices | Mean WER without runaway takes | 0.053 | 0.042 | — |
| Unseen voices | Runaway takes | 4 / 300 | 0 / 300 | 0.12 |
| Unseen voices | Takes flagged too long | 7 / 300 | 3 / 300 | 0.34 |
| Unseen voices | Short prompts with CER > 0.5 | 9 / 45 | 2 / 45 | 0.05 |
| Target speaker | Short prompts with CER > 0.5 | 1 / 9 | 0 / 9 | 1.00 |

**Conclusion.** The pre-registered decision is only partly evaluated: on unseen voices every non-regression gate
passes (`passed: true` in `ab_report_base.json`), but on 300 of the pre-registered 900 pairs, so the decision
stays provisional until the remaining pairs are measured. The per-sample fix is retained either way, because F2's
demonstrating test shows it is correct regardless of its measured effect. What the data shows about the size of the effect
is narrower than the means suggest. Per-sample masking produced fewer badly failed short prompts on unseen voices
(9 vs 2 of 45, $p = 0.05$) and no runaway takes (4 vs 0, $p = 0.12$); outside those takes, intelligibility is
about the same (mean WER 0.053 vs 0.042, medians 0). This is consistent with F2 causing occasional runaway
generation, especially on short prompts, but at $N=300$ it is suggestive, not established. In the target regime
there is no detectable intelligibility difference, and the SECS gate **fails**: similarity to the target speaker
is 0.017 lower with the fix (95% CI $[-0.030, -0.004]$, margin 0.02). Establishing the effect on runaway
generation needs the pre-registered 900 pairs, which requires re-evaluating the existing checkpoints, not
retraining.

### 6.1.1 Loss Decomposition and The Evaluation-Loss Metric

During fine-tuning, upstream masking appeared to yield slightly lower overall evaluation loss (3.309 vs. 3.424 in the trainer's batch-averaged eval loss). To analyze this behavior, we conducted a token-level loss decomposition on the held-out evaluation clips. Both checkpoints were evaluated on identical validation batches ($B=4$, comprising Batch 0 with 423 valid tokens and Batch 1 with 387 valid tokens, totaling $N=810$ active tokens). Cross-entropy loss was calculated directly from raw token-level predictions across two mutually exclusive partitions:
1. **Conditioning Prompt Tokens ($\mathcal{L}_{\text{prompt}}$):** Target code positions $[lo_i, hi_i)$ corresponding to the reference voice prompt prefix provided in context ($N=559$ tokens across both batches).
2. **Speech Generation Tokens ($\mathcal{L}_{\text{gen}}$):** The non-prompt speech codes representing autoregressive speech generation outside the conditioning span ($N=251$ tokens across both batches).

**Table 7d.** Token-Level Cross-Entropy Loss Decomposition on Held-Out Validation Clips ($B=4$, $N=810$ Total Active Tokens).

| Evaluation Component | Upstream Masking (F2 Defect) | Per-Sample Masking (Bugfix) | $\Delta$ (Bugfix − Upstream) | Notes / Reduction Method |
| :--- | :---: | :---: | :---: | :--- |
| **Prompt Tokens ($\mathcal{L}_{\text{prompt}}$, $N=559$)** | **3.170** | 3.372 | +0.202 | $\sum \text{CE} / 559$ (1771.89 / 559 vs. 1885.16 / 559) |
| **Generation Tokens ($\mathcal{L}_{\text{gen}}$, $N=251$)** | 3.550 | **3.470** | **−0.080** | $\sum \text{CE} / 251$ (890.98 / 251 vs. 871.09 / 251) |
| **Token-Weighted Aggregate ($\mathcal{L}_{\text{all}}$, $N=810$)** | **3.288** | 3.403 | +0.115 | $\sum \text{CE} / 810$ (2662.87 / 810 vs. 2756.25 / 810) |
| **Trainer Aggregate Loss ($\mathcal{L}_{\text{eval}}$)** | **3.309** | 3.424 | +0.115 | Unweighted batch mean $(\mathcal{L}_{\text{batch } 0} + \mathcal{L}_{\text{batch } 1}) / 2$ |

These measurements clarify the relationship between prompt masking and evaluation loss:
- **Trainer Reduction Mechanics:** In Coqui's `GPTTrainer.eval_step`, conditioning masking is explicitly bypassed via `batch["cond_idxs"] = None` (`# ignore masking for more consistent evaluation`). In `GPT.forward`, when `cond_idxs is None`, the conditioning masking block is skipped and only sequence padding (`mel_targets[idx, l+1:] = -1`) is ignored; prompt tokens receive standard loss targets without any other masking. Each batch computes mean cross-entropy over its valid tokens, and the trainer's `KeepAverage` reducer averages batch losses equally. For Upstream, Batch 0 (423 tokens, loss 3.327) and Batch 1 (387 tokens, loss 3.292) average to 3.309; for Per-Sample, Batch 0 (423 tokens, loss 3.456) and Batch 1 (387 tokens, loss 3.392) average to 3.424.
- **Partition Trade-off:** The upstream model exhibits lower prompt-region loss ($3.170$ vs. $3.372$) but higher generation-region loss ($3.550$ vs. $3.470$). This observation is consistent with the hypothesis that unmasking prompt-region targets for samples $0 \dots N-2$ during upstream training allowed the model to optimize against prompt tokens present in the conditioning prefix. Under per-sample masking, the conditioning-prompt region is excluded from the loss for every sample; the fixed model yields lower loss on the remaining generation-region tokens ($3.470$ vs. $3.550$).
- **Metric Confounding:** In the evaluation set, prompt tokens account for 69.0% of all active tokens (559 of 810 tokens). Because upstream evaluation similarly disables prompt masking (`cond_idxs = None`), prompt tokens dominate the unmasked aggregate metric. The lower prompt-region loss pulls the overall unmasked loss down to 3.288 (and 3.309 under batch-level reduction), creating the appearance of lower evaluation loss despite higher loss on actual speech generation. When evaluation is restricted strictly to non-prompt generation tokens ($\mathcal{L}_{\text{gen}}$), the per-sample model demonstrates superior predictive performance.

With A3 run, the remaining defaults (A1, A2, A4–A8) rest on the peer-reviewed literature summarized in Table 6 and Section 3.3. They have not been tested on this model, so they stay at evidence level E2/E3.

## 7 Discussion and limitations

**What has been shown.** The defects in Table 1 are properties of the code and the released checkpoint. They are
established by tests rather than by measurement noise. F1 and the drift measurement together show that the
DVAE–GPT code interface is fragile. The community consensus that the DVAE should not be trained has the right
default for the wrong reason: the reason is coupling, not universality.

**What has not been shown.** Full language adaptation across non-Latin scripts has not yet been evaluated end-to-end. A3 shows the F2 fix passes its non-regression gates and suggests, without establishing, that F2 causes occasional runaway generation on short prompts (§6.1). Ablations A1, A2 and A4–A8 have not been run.

**Limitations.**

1. **Smoke-scale training.** All training results come from eight clips. They establish that the stages work,
   not how well.
2. **Alignment thresholds.** The validation thresholds rest on one English recording.
3. **Baseline language coverage.** The baseline now covers 900 utterances across five languages (`en`, `es`, `fr`, `de`, `hi`), expanded from the earlier 120-utterance English pilot; the remaining 12 languages supported by XTTS-v2 have not yet been evaluated.
4. **Automatic metrics only.** Whisper-based WER/CER inherits the ASR model's biases. UTMOS was trained on
   English listening tests and is used only relative to the baseline. SECS is unreliable below about 2 s. No
   human listening test has been run; *[a MUSHRA or MOS study is planned]*.
5. **Gate margins.** The gate margins are judgements (E5) sized against observed seed noise. They are not derived
   from perceptual thresholds.
6. **Licensing.** The XTTS-v2 weights are under the Coqui Public Model License, and the MMS forced-alignment
   model under CC BY-NC 4.0. Results apply to non-commercial use of these components.

**Threats to validity.** Our audit covers the training path we use; untested code paths may contain further
defects. Every external fact in the repository (hashes, timestamps, transcripts) comes from recorded command
output, but the Colab runs used uploaded working trees rather than tagged commits. Runs 2 and 4 recorded a content
fingerprint of the code; run 4's equals that of commit `1838b4c`, which identifies its code exactly. Runs 1 and 3
are identified only by date. Run 4 also predates source pinning, so the digests of its input audio were not
recorded.

## 8 Conclusion

We audited the XTTS-v2 fine-tuning pipeline and found six defects that silently change what is trained, the
four in the training code present since the original release, and a seventh, introduced by the maintained fork,
that silently rewrites the released weights file. We built a three-stage recipe in which every decision carries its evidence
level and every repair is guarded by a test, together with a data pipeline whose forced-alignment validator
detects misplaced cuts on real audio. We also built an evaluation protocol designed so that a fine-tune can be
judged only against pre-registered gates on a frozen test set. Measurements so far show that the stock baseline is deterministic per seed within an environment and agrees within intervals across environments, DVAE fine-tuning breaks the code interface the pretrained GPT depends on, and Stage 2 GPT fine-tuning adapts to a target speaker on held-out speech (all five target gates pass) at a measurable cost to unseen speakers (the pre-registered preservation SECS gate fails). The planned multi-language experiments will establish which of the recipe's choices improve cross-lingual synthesis. *[To be completed with A1–A8.]*

---

## Reproducibility statement

All code, tests, evaluation definitions, pinned hashes and archived run evidence are in this repository.

- **Artifacts:** `artifacts.lock.json` lists every model artifact and its hash.
- **Test set and voices:** `eval/dataset/eval_set.json` is the frozen test set, and
  `eval/dataset/reference_voices.lock.json` pins the reference voices.
- **Gates:** `eval/gates.json` holds the pre-registered gates.
- **Run evidence:** `docs/evidence/` holds the archived evidence of each Colab run, with environment snapshots.
- **Not in the repository:** target-speaker clips and their lock (`eval/target_voices/`) are kept private by
  design, because they hold a real person's voice, transcripts and local paths. Target-speaker results can be
  reproduced only by whoever holds that speaker's recordings; everything else here can be rerun from public
  data.
- **Notebook:** `notebooks/phase0_phase1_colab.ipynb` reproduces the GPU steps.

## Data and licences

- **Evaluation voices:** CMU ARCTIC; each speaker's licence file is pinned by hash.
- **Smoke-run data:** LJSpeech and the JFK inaugural address, both public domain.
- **Long-audio example:** a LibriVox recording of *The Art of War*, public domain.
- **Models:** the XTTS-v2 weights (Coqui Public Model License) and MMS_FA and ECAPA2 (CC BY-NC 4.0) are
  downloaded by the user under their own licences and are not redistributed.

## Ethics statement

Voice cloning can be misused for impersonation and fraud. The target-speaker tooling assumes the speaker has
consented to being cloned. Target-speaker audio is kept out of version control by default. We release no
cloned voices of real people. Evaluation uses CMU ARCTIC speakers, recorded for speech-synthesis research.

## Author contributions and AI-assistance disclosure

*[CRediT roles to be added.]* Parts of the code, tests and this draft were written with an AI coding assistant
(Claude, Anthropic). All results were produced by running the code. Each number is traceable to an archived
output, and every claim was checked against the repository by the author.

## Acknowledgements

*[To be added.]*

---

## References

[1] E. Casanova, K. Davis, E. Gölge, G. Göknar, I. Gulea, L. Hart, A. Aljafari, J. Meyer, R. Morais, S. Olayemi,
and J. Weber, "XTTS: A massively multilingual zero-shot text-to-speech model," in *Proc. Interspeech*, 2024, pp. 2480–2484.
arXiv:2406.04904.

[2] J. Betker, "Better speech synthesis through scaling," arXiv:2305.07243, 2023.

[3] J. Kong, J. Kim, and J. Bae, "HiFi-GAN: Generative adversarial networks for efficient and high fidelity
speech synthesis," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020, pp. 17022–17033.
arXiv:2010.05646.

[4] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," in *Proc.
Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 6306–6315.
arXiv:1711.00937.

[5] C. Zheng and A. Vedaldi, "Online clustered codebook," in *Proc. IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, pp. 22744–22753.
arXiv:2308.12152.

[6] K. Dobler and G. de Melo, "FOCUS: Effective embedding initialization for monolingual specialization of
multilingual models," in *Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2023, pp. 13440–13454.
arXiv:2305.14481.

[7] B. Minixhofer, F. Paischer, and N. Rekabsaz, "WECHSEL: Effective initialization of subword embeddings for
cross-lingual transfer of monolingual language models," in *Proc. Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)*, 2022, pp. 3992–4006.
arXiv:2112.06598.

[8] M. Yang, S. Ding, T. Chen, T. Wang, and Z. Wang, "Towards lifelong learning of multilingual text-to-speech
synthesis," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2022, pp. 7447–7451.
arXiv:2110.04482.

[9] E. Casanova, J. Weber, C. Shulby, A. C. Junior, E. Gölge, and M. A. Ponti, "YourTTS: Towards zero-shot
multi-speaker TTS and zero-shot voice conversion for everyone," in *Proc. 39th International Conference on Machine Learning (ICML)*, 2022, pp. 2709–2720.
arXiv:2112.02418 (including erratum).

[10] A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, "Connectionist temporal classification: Labelling
unsegmented sequence data with recurrent neural networks," in *Proc. 23rd International Conference on Machine Learning (ICML)*, 2006, pp. 369–376.

[11] V. Pratap, A. Tjandra, B. Shi, P. Tomasello, A. Babu, S. Kundu, A. Elkahky, Z. Ni, A. Vyas, M. Fazel-Zarandi,
A. Baevski, Y. Adi, X. Zhang, W. N. Hsu, A. Conneau, and M. Auli, "Scaling speech technology to 1,000+ languages," *Journal of Machine Learning Research (JMLR)*,
vol. 25, no. 97, pp. 1–52, 2024.
arXiv:2305.13516.

[12] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust speech recognition via
large-scale weak supervision," in *Proc. 40th International Conference on Machine Learning (ICML)*, 2023, pp. 28492–28518.
arXiv:2212.04356.

[13] J. Thienpondt and K. Demuynck, "ECAPA2: A hybrid neural network architecture and training strategy for
robust speaker embeddings," in *Proc. IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)*, 2023, pp. 1–8.
arXiv:2305.08500.

[14] T. Saeki, D. Xin, W. Nakata, T. Koriyama, S. Takamichi, and H. Saruwatari, "UTMOS: UTokyo-SaruLab system for
VoiceMOS Challenge 2022," in *Proc. Interspeech*, 2022, pp. 4521–4525.
arXiv:2204.02152.

[15] B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. Boca Raton: Chapman & Hall/CRC, 1993.

[16] J. Pineau, P. Vincent-Lamarre, K. Sinha, V. Larivière, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and
H. Larochelle, "Improving reproducibility in machine learning research," *Journal of Machine Learning Research (JMLR)*,
vol. 22, no. 164, pp. 1–20, 2021.

[17] J. Kominek and A. W. Black, "The CMU Arctic speech databases," in *Proc. 5th ISCA Speech Synthesis
Workshop (SSW5)*, 2004, pp. 223–224.

[18] K. Ito and L. Johnson, "The LJ Speech dataset," 2017. https://keithito.com/LJ-Speech-Dataset/

[19] Y. Chen, S. Chen, H. Wu, Z. Yao, Z. Chen, P. Wang, S. Zhao, L. He, T. Hayashi, and J. Li,
"VALL-E 2: Neural codec language models are human parity zero-shot text to speech synthesizers," arXiv:2406.05370, 2024.

[20] A. S. Poerner, U. Hennig, and H. Schütze, "Evaluating subword embedding initialization for monolingual language transfer," in *Proc. 28th Conference on Computational Natural Language Learning (CoNLL)*, 2024, pp. 1–16.

[21] Y. Gu, X. Yin, and Z. Ma, "Evaluating speaker encoders for zero-shot text-to-speech," arXiv:2506.20190, 2025.

[22] B. Ebing, L. Keller, and G. Glavaš, "One script instead of hundreds? On pretraining romanized encoder language models," arXiv:2601.05776, 2026.

---

## Appendix A: Claim-to-evidence map

| Claim (section) | Evidence in this repository |
|---|---|
| F7: the fork's loader rewrites `model.pth` (3.2, 5.1) | `tests/test_checkpoint_loading.py` (upstream rewrites a legacy checkpoint; the guarded loader doesn't; every loader applies the guard); `docs/evidence/upstream/coqui_tts_v0.22.0_check.md` (absent in v0.22.0) |
| F1–F6 exist in the pinned fork (3.2, 5.1) | `docs/recipe_design.md` §2; tests named there, e.g. `test_upstream_f2_masks_only_the_last_sample_in_the_wrong_units`, `test_new_text_rows_mean_init_and_upstream_f6`, `test_whisper_word_timestamps_have_no_gaps` |
| Findings present in coqui-ai/TTS v0.22.0 (5.1) | `docs/evidence/upstream/coqui_tts_v0.22.0_check.md` |
| Artifact hashes, mel-stats identity (5.2) | `artifacts.lock.json`; `docs/evidence/artifact_verification.json`; `docs/evidence/colab_run1/`, `colab_run2/` |
| Bit-identical reruns; seed variance (5.2) | `docs/evidence/colab_run1/pilot_en_reproducibility.json`; `docs/evidence/colab_run2/determinism_check.json` |
| Baseline, Table 2 (5.3) | `eval/baselines/base_xtts_s12_index.json`, `docs/evidence/run5/README.md` |
| Cross-environment rerun: 852/900 identical takes; Spanish runaway take (5.2) | `docs/evidence/phase8_reproducibility/README.md`, `docs/evidence/phase8_reproducibility/reproducibility_report.json` |
| Planner regression (5.4) | `docs/evidence/colab_run3/README.md`; `tests/test_prepare.py` (`test_cuts_prefer_the_longest_pause_jfk`, `test_pause_never_outweighs_speech`, `test_equal_speech_plans_cut_at_the_longer_pause`) |
| Shifted-cut word scores, Table 3 (5.4) | `docs/evidence/colab_run4/jfk_realign_shift_*.json`; `test_mms_jfk_alignment_and_shifted_cut` |
| Multilingual alignment validation, Table 3b (5.4) | `docs/evidence/phase6/README.md`; `docs/evidence/phase6/multilingual_alignment_validation.json`; `docs/evidence/phase6/ALIGNMENT_VALIDATION_METHODOLOGY.md` |
| Held-out alignment validation on unseen speakers & languages (5.4) | `docs/evidence/phase6_heldout/README.md`; `docs/evidence/phase6_heldout/heldout_alignment_validation.json`; `docs/evidence/phase6_heldout/heldout_alignment_validation_rerun.json`; `docs/evidence/phase6_heldout/heldout_alignment_validation_run3.json`; `docs/evidence/phase6_heldout/source_pins.json` |
| Smoke runs and DVAE drift, Table 4 (5.5) | `docs/evidence/colab_run4/` (`stage1_smoke.json`, `training_summary.json`, `runs_gpt_smoke_log.txt`, `stage3_metrics.jsonl`); drift defined in `xtts_trainer/stage1_dvae/train.py` (`evaluate_dvae`) |
| Paired A/B and gates (3.5) | `scripts/compare_ab.py`; `eval/gates.json`; `tests/test_compare_ab.py` |
| Target-speaker hold-out guard (3.5) | `eval/target_voices.py`; `xtts_trainer/data/validate.py`; `tests/test_target_voices.py` |
| Preservation on unseen voices, Table 5b (§5.6, Run 6): SECS gate failed | `eval/gates.json` (pre-registered, `603e0d3`); `docs/evidence/run6_target_speaker/ab_report_preservation.json`; `eval/baselines/base_xtts_s20_en.json`; `docs/evidence/run6_target_speaker/tuned_xtts_s20_arctic_en.json` |
| Target speaker adaptation, Table 5 (§5.6, Run 6) | `eval/gates_target.json`; `docs/evidence/run6_target_speaker/ab_report_target.json`; `docs/evidence/run6_target_speaker/evidence/prep_report.json`; `docs/evidence/run6_target_speaker/evidence/validation_report.json`; `docs/evidence/run6_target_speaker/evidence/training_summary.json` |
| Source pinning (3.6) | `xtts_trainer/speech_data.py`; `docs/evidence/source_pins.md` (SHA-256 pins, with the SHA-1 cross-check) |
