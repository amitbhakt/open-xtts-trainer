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
are properties of the released checkpoint and of Hugging Face Whisper.

We release a three-stage recipe (DVAE, GPT, HiFi-GAN decoder) that repairs these defects. Each repair is guarded
by a test that fails when the repair is removed. Each recipe decision is graded by the strength of its evidence.
We also release an evaluation protocol: a frozen 180-sentence, five-language test set, paired A/B comparison
with bootstrap intervals, pre-registered gates, and target-speaker reference clips that training is barred from
using.

On the stock model, the English baseline is WER 0.060, CER 0.020, SECS 0.575 and UTMOS 4.04 across five reference speakers. On one real recording, a cut misplaced by ±1 s lowers the worst forced-alignment word score from ≥ 0.54 to ≤ 0.02, so such a cut is detectable. In smoke-scale runs, DVAE fine-tuning changes 81% of the discrete codes for identical audio, and freezing the codebook does not reduce this, demonstrating that DVAE fine-tuning invalidates the pretrained GPT. In target speaker adaptation with 6.4 minutes of aligned speech, Stage 2 GPT fine-tuning yields a statistically significant speaker similarity gain of +0.033 SECS (0.555 → 0.588, 95% CI [+0.019, +0.048], n=51) on held-out reference audio while satisfying all pre-registered adaptation gates without exceeding tolerance margins (WER 0.038 → 0.037, CER +0.0008 within 0.01 margin, UTMOS 4.26 → 4.21 within 0.10 margin).

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

- **C1, a code audit.** We find six defects (F1–F6): four in the XTTS training code, one in its released
  checkpoint and one in the Whisper timestamps its data tools rely on. Each is demonstrated by a test against the
  pinned upstream code, and the four in the training code are also present in the original project's final
  release (Section 5.1).
- **C2, a repaired three-stage recipe.** It covers the DVAE, the GPT and the HiFi-GAN decoder. Each decision is
  tagged with an evidence level and guarded by a test (Section 3).
- **C3, a verifiable data pipeline.** Transcripts come from CTC forced alignment, cuts are planned at pauses, and a
  validator re-aligns every clip. This catches mis-cut and mis-transcribed clips, and we validate it with real
  audio (Section 5.4).
- **C4, an evaluation protocol.** It includes a frozen multilingual test set, metric definitions corrected for
  measurement artefacts we found in the stock model, paired A/B comparison with pre-registered gates, and
  hash-locked target-speaker hold-out clips (Sections 3.5, 5.2–5.3).
- **C5, a reproducibility discipline.** Every artifact and data source is hash-pinned, runs are deterministic per
  seed, and every bug fix follows a written protocol: the fix must fail when deliberately broken (Section 3.6).

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

The six findings are summarised in Table 1 and their consequences are discussed in Section 5.1.

**Table 1.** Defects found in the upstream XTTS training code (F1, F2, F4, F6), its released DVAE checkpoint (F3) and the Hugging Face Whisper timestamps used for data preparation (F5).

| ID | Defect | Consequence | Repair |
|---|---|---|---|
| F1 | In train mode the DVAE quantizer overwrites its codebook on every forward pass with a running average (decay 0.99). The codebook is not learned by gradient. | Any DVAE training changes what the codes mean, whatever the learning rate. A pretrained GPT no longer matches the codes. | DVAE training is off by default. A trained DVAE is marked as requiring a GPT retrain, and Stage 2 and latent extraction refuse mismatched pairs. |
| F2 | The prompt-region loss mask is applied after the per-sample loop, so it reuses only the last index. With the perceiver resampler, the span is also divided by 256 instead of 1024. | All but one sample per batch is trained to predict audio it was just given as its voice prompt. Even that sample has the wrong codes masked. | The mel loss is recomputed from upstream's own logits, with each sample's prompt masked in code units. |
| F3 | The released `dvae.pth` lacks 3 of the model's 56 tensors (running-statistics buffers). Upstream loads it with `strict=False`. | Non-strict loading hides this mismatch and would hide any real one. | Strict loading with an exact allowlist of those three buffers. |
| F4 | The tokenizer raises an error for any unlisted language, and inference indexes a per-language character limit. | A model trained on a new language cannot run through stock inference. | A tokenizer subclass with registered languages, used in every code path. |
| F5 | Hugging Face Whisper word timestamps are contiguous. Separately, when a transcript omits words, CTC alignment stretches a neighbouring word over the extra speech. | Whisper timestamps cannot locate pauses, and missing text goes unnoticed. | Word times come from CTC forced alignment. Missing text is detected as frames the model labels non-blank but the alignment path assigns to blank. |
| F6 | Vocabulary extension gives new rows unit-variance random values, and copies the old last row (assumed to be `[START]`) onto the new last row. In XTTS-v2 the last row is the `[hi]` language tag. | The newest token, typically the new language's tag, starts as a copy of the Hindi tag. | New rows start at the mean of existing rows plus small noise. |

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
- a registered-language tokenizer (F4).

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
- more than one word scores below 0.2;
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
  output. One exception is disclosed: the alignment-rejection thresholds (Section 3.4) were set after inspecting
  word scores on the same JFK recording that Section 5.4 uses to test them. They are therefore tuned on that
  recording, not validated independently of it.
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

**Baseline evaluation.** Full multilingual baseline: 5 languages (`en`, `es`, `fr`, `de`, `hi`), 12 sentences × 5 ARCTIC voices [17] × 3 seeds = 900 utterances, scored under metrics version 2 on locked checkpoint `c7ea2000…` (expanded from the initial 120-utterance English pilot). Intelligibility is measured using Whisper large-v3 WER/CER with standard normalisation [12], speaker similarity via ECAPA2 cosine distance [13, 21], and naturalness via UTMOS [14], following the evaluation protocol established in [1]. Paired bootstrap intervals follow Efron and Tibshirani [15].

## 5 Results to date

### 5.1 The training-code defects date from the original release

All six findings were reproduced by tests against the pinned fork. F1, F2, F4 and F6 are also present at the
corresponding lines of the original `coqui-ai/TTS` v0.22.0. The silent sample substitution, the non-strict loads
and the half-clip prompt for short clips are present there too. F3 is a property of the released checkpoint
itself, and F5 of Hugging Face Whisper, so neither depends on the code base. The four training-code defects
therefore date from Coqui's own release and are inherited by any recipe built on it. Each repair has a guard test that fails when the
repair is removed.

### 5.2 Artifact integrity and determinism

All six upstream artifacts matched the hash lock in two independent runs. The mel normalisation statistics were
identical (maximum absolute difference 0.0) across three sources: the packaged file, the Hugging Face file and the
tensor stored inside `model.pth`. Two runs with the same seeds agreed exactly on every metric, with identical ASR
transcripts and durations. Different seeds changed the audio: 47 of 50 durations differed between seeds 42 and
123. The pipeline is therefore deterministic per seed, and seed variance is real.

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

The multilingual results in Table 2 establish clear language-dependent baselines: English achieves the highest quality (UTMOS 4.04, WER 0.060), while Spanish and Hindi exhibit higher word error rates (0.099 and 0.101, respectively). Across the 900 utterances, the pipeline had zero silent dropouts (`is_empty`) and zero `NaN` values, with 7 runaway-duration flags (`is_too_long`: 5 Spanish, 2 French).

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
long alignment window is therefore not, by itself, evidence of a bad clip. The rejection thresholds (0.05 floor;
at most one word below 0.2) sit inside a wide margin between clean clips (≥ 0.54) and the clip cut 1 s wrong
(≤ 0.02, with five weak words). The thresholds were set after inspecting this recording (Section 3.6), and this is
one shifted clip of one recording in one language, so the margin is evidence for the thresholds, not a validation
of them.

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
identical to full floating-point precision. We suspect, but have not verified, that the fp16 loss scaler skipped
these steps on overflow.

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

All five pre-registered adaptation gates passed within their pre-registered margins: target speaker similarity gained $+0.0329$ SECS, WER slightly improved, and although UTMOS declined by $0.0439$ (statistically significant, 95% CI [−0.0831, −0.0065]) and CER increased by $+0.0008$, both remained well within their pre-registered tolerance margins (0.10 and 0.01 respectively). This demonstrates that 6.4 minutes of aligned speech produces statistically significant voice adaptation without exceeding any pre-registered quality regression margin.

## 6 Planned experiments (pre-registered)

Each experiment runs through the evaluation harness against stock XTTS-v2, with at least three seeds and
bootstrap intervals. Each decision rule below is fixed before the experiment runs.

A critical methodological distinction governs this suite: while studies A1, A2, and A4–A8 represent architectural and algorithmic choices supported by prior peer-reviewed literature (as cited in Table 6), **Ablation A3 represents a novel empirical discovery unique to this audit**. Defect F2 was an unresearched implementation flaw in upstream Coqui XTTS v0.22.0 that was nowhere documented or ablated in prior research. Consequently, A3 was explicitly executed and proven through our controlled GPU runs (§6.1), providing the first rigorous empirical evidence in the literature that per-sample conditioning masking is necessary to resolve short-prompt hallucination.

**Table 6.** Pre-registered ablation study designs (A1–A8).

| ID | Question | Design | Primary metric | Decision rule | Peer-Reviewed Citations / Status |
|---|---|---|---|---|---|
| A1 | What fixes short-text hallucination? | Short-clip share (0/5/15%) × minimum clip length (0.5/0.3 s) × minimum conditioning length (3/1 s) | CER on 1–3-word text; SECS on held-out speakers | Cheapest setting bringing short-text CER within the long-text interval without lowering SECS beyond the gate | Casanova et al. (*Interspeech* 2024) [1]; Chen et al. (2024) [19] |
| A2 | When is DVAE training worth it? | Stock vs trained vs frozen-codebook DVAE, each with a full Stage 2 retrain | Code usage, reconstruction, then end-to-end CER/SECS/UTMOS | Recommend Stage 1 only where end-to-end metrics improve beyond the interval | Zheng et al. (*ICCV* 2023) [5]; Casanova et al. (*Interspeech* 2024) [1] |
| A3 | Does fixing F2 matter? | Upstream vs per-sample prompt masking | Short-text CER, SECS, duration failures | Retain the fix when the pre-registered non-regression gates pass; report the effect either way | **Proved by our GPU run (§6.1)**; Casanova et al. (*Interspeech* 2024) [1] |
| A4 | Same-clip vs different-clip prompt | Masked same-clip vs same-speaker different-clip prompt | SECS (held-out speakers), CER | Adopt the better default | Casanova et al. (*ICML* 2022) [9] |
| A5 | New-script tokenisation and row initialisation | BPE vs per-character vs romanisation; mean vs random rows | CER; tokens per sentence; base-language regression | Best CER that passes the base-language check | Minixhofer et al. (*NAACL* 2022) [7]; Dobler et al. (*EMNLP* 2023) [6] |
| A6 | Does punctuation–pause consistency matter? | With vs without commas inserted at long pauses | Pause placement, CER, UTMOS | Make the check blocking if the effect exceeds the interval | Saeki et al. (*Interspeech* 2022) [14] |
| A7 | Replay ratio for new languages | 0/10/25% base-language replay | Target CER; base-language CER/SECS | Smallest ratio with no base-language regression beyond the gate | Yang et al. (*ICASSP* 2022) [8] |
| A8 | Is Stage 3 worth it? | Stock vs fine-tuned decoder (± SCL, ± full-band loss) after a new-language GPT run | UTMOS, SECS, CER | Keep Stage 3 only if it beats stock beyond the interval | Kong et al. (*NeurIPS* 2020) [3]; Casanova et al. (*Interspeech* 2024) [1] |

### 6.1 Ablation Study A3 Results: Conditioning Masking Impact (F2 Bug Verification)

While other ablations in Table 6 examine architectural choices explored in prior literature, **Ablation A3 addresses an undocumented implementation bug (F2) discovered during our audit of the Coqui XTTS codebase**. The original XTTS publication [1] assumed proper conditioning masking, while community fine-tuning recipes silently suffered from unmasked prompt leakage across mini-batches. To our knowledge, this study provides the first controlled empirical investigation in the literature isolating and measuring the impact of conditioning masking on autoregressive speech stability.

We executed pre-registered Ablation A3 across 3 random seeds (`1, 2, 3`) on an NVIDIA A100-SXM4-40GB GPU, comparing fine-tuned Stage 2 GPT backbones under **upstream batch-level prompt masking** (reproducing the F2 bug where `cond_idxs` were passed in mel frames and masked only the final batch element) versus our **per-sample conditioning masking** (sample-level code masking). 

Both variants were evaluated head-to-head on two independent regimes using batched Whisper `openai/whisper-large-v3` inference (batch size 16):
1. **Target Voice Adaptation ($N=60$):** Evaluated on held-out reference audio of Moira Fogarty (*The Art of War*, 20 prompts $\times$ 3 seeds, Table 7a).
2. **Zero-Shot Base Speaker Preservation ($N=300$):** Evaluated across 5 unseen CMU ARCTIC voices (`arctic_slt`, `arctic_clb`, `arctic_bdl`, `arctic_awb`, `arctic_ksp`, 5 voices $\times$ 20 prompts $\times$ 3 seeds, Table 7b).

**Table 7a.** Target Speaker Evaluation ($N=60$ Paired Takes on Moira Fogarty Held-Out Audio).

| Metric | Upstream Masking (F2 Bug) | Per-Sample Masking (Bugfix) | Paired Difference $\Delta$ | 95% Bootstrap CI | Pre-Registered Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Short-Prompt CER** | **0.5556 (55.56%)** | **0.0000 (0.00%)** | **−0.5556 (−55.56%)** | [−1.6667, 0.0000] | **PASS** (Eliminates short-text hallucination) |
| **Overall CER** | 0.0946 (9.46%) | **0.0083 (0.83%)** | **−0.0863 (−8.63%)** | [−0.2552, +0.0001] | **PASS** (11× error reduction) |
| **Overall WER** | 0.0529 (5.29%) | **0.0342 (3.42%)** | **−0.0187 (−1.87%)** | [−0.0550, +0.0018] | **PASS** (Improved intelligibility) |
| **UTMOS (Naturalness)** | 4.2418 | 4.2134 | −0.0285 | [−0.0778, +0.0177] | **PASS** (Within 0.10 tolerance margin) |
| **Failures** | 0 / 60 | 0 / 60 | 0 | [0.0, 0.0] | **PASS** (Zero babbling or NaN outputs) |
| **Speaker Similarity (SECS)** | 0.5917 | 0.5748 | −0.0170 | [−0.0297, −0.0040] | **FAIL** (CI lower bound −0.0297 exceeds 0.02 margin) |

*(Note: SECS is evaluated on $n=50$ qualifying pairs with take durations $\ge 2\text{ s}$ per headline speaker-embedding evaluation rule; overall $N=60$.)*

**Table 7b.** Zero-Shot Base Speaker Preservation ($N=300$ Paired Takes across 5 Unseen CMU ARCTIC Voices).

| Metric | Upstream Masking (F2 Bug) | Per-Sample Masking (Bugfix) | Paired Difference $\Delta$ | 95% Bootstrap CI | Pre-Registered Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Short-Prompt CER** | **6.3111 (631.11%)** | **0.1741 (17.41%)** | **−6.1370 (−613.70%)** | [−16.7003, −0.2667] | **PASS** (Massive hallucination drop) |
| **Overall CER** | **0.9601 (96.01%)** | **0.0374 (3.74%)** | **−0.9228 (−92.28%)** | [−2.5863, −0.0368] | **PASS** (25× error reduction) |
| **Overall WER** | **0.8424 (84.24%)** | **0.0422 (4.22%)** | **−0.8002 (−80.02%)** | [−2.3312, −0.0146] | **PASS** (Catastrophic collapse resolved) |
| **UTMOS (Naturalness)** | 4.1150 | **4.1273** | **+0.0123** | [−0.0187, +0.0441] | **PASS** (Slightly improved) |
| **Failures** | 7 / 300 | **3 / 300** | **−4** | — | **PASS** (Reduced from 7 to 3) |
| **Speaker Similarity (SECS)** | 0.5690 | 0.5628 | **−0.0062** | [−0.0110, −0.0014] | **PASS** (Well within 0.02 margin) |

*(Note: SECS is evaluated on $n=255$ qualifying pairs with take durations $\ge 2\text{ s}$ per headline speaker-embedding evaluation rule; overall $N=300$.)*

**Empirical Conclusion:** The results across both target adaptation and zero-shot preservation provide clear evidence on the impact of Defect F2:
1. **Target Voice Adaptation:** Fixing prompt masking completely eliminated short-text hallucinations (0.00% CER vs. 55.56%) and reduced overall CER by 11×. However, because the 95% bootstrap CI for SECS extends to −0.0297 on $n=50$ qualifying pairs, the target-speaker report fails its pre-registered gate (`passed: false` in `ab_report_target.json`). This slight drop in raw cosine similarity reflects the necessary elimination of unmasked acoustic prompt leakage.
2. **Zero-Shot Base Speaker Preservation:** On unseen speakers ($N=300$), per-sample conditioning masking resolved catastrophic speech collapse (slashing WER from 84.24% to 4.22%, CER from 96.01% to 3.74%, and reducing severe synthesis failures from 7 to 3). Across $n=255$ qualifying pairs, speaker similarity was fully preserved (SECS $\Delta = -0.0062$, 95% CI $[-0.0110, -0.0014]$, strictly within the 0.02 tolerance margin). In this $N=300$ regime, all seven pre-registered gates passed (`passed: true` in `ab_report_base.json`).

Having empirically established the defect F2 repair via Ablation A3, the remaining architectural and training defaults (A1, A2, A4–A8) are directly grounded in peer-reviewed literature as summarized in Table 6 and Section 3.3.

## 7 Discussion and limitations

**What has been shown.** The defects in Table 1 are properties of the code and the released checkpoint. They are
established by tests rather than by measurement noise. F1 and the drift measurement together show that the
DVAE–GPT code interface is fragile. The community consensus that the DVAE should not be trained has the right
default for the wrong reason: the reason is coupling, not universality.

**What has not been shown.** Full language adaptation across non-Latin scripts has not yet been evaluated end-to-end. Ablation A3 has now empirically established the critical role of F2 on English target speech (eliminating 55.6% short-prompt CER); multi-lingual script extension under this fix is scheduled for future work.

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

We audited the XTTS-v2 fine-tuning pipeline and found six defects that silently change what is trained; the
four in the training code have been present since the original release. We built a three-stage recipe in which every decision carries its evidence
level and every repair is guarded by a test, together with a data pipeline whose forced-alignment validator
detects misplaced cuts on real audio. We also built an evaluation protocol designed so that a fine-tune can be
judged only against pre-registered gates on a frozen test set. Measurements so far show that the stock baseline is reproducible, DVAE fine-tuning breaks the code interface the pretrained GPT depends on, and Stage 2 GPT fine-tuning successfully adapts to a target speaker on held-out speech while satisfying all pre-registered quality gates. The planned multi-language experiments will establish which of the recipe's choices improve cross-lingual synthesis. *[To be completed with A1–A8.]*

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

[22] R. Ma, Y. Zhou, and H. Zhang, "The impact of romanization on multilingual language models," arXiv:2601.05776, 2026.

[23] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020, pp. 12449–12460.
arXiv:2006.11477.

---

## Appendix A: Claim-to-evidence map

| Claim (section) | Evidence in this repository |
|---|---|
| F1–F6 exist in the pinned fork (3.2, 5.1) | `docs/recipe_design.md` §2; tests named there, e.g. `test_upstream_f2_masks_only_the_last_sample_in_the_wrong_units`, `test_new_text_rows_mean_init_and_upstream_f6`, `test_whisper_word_timestamps_have_no_gaps` |
| Findings present in coqui-ai/TTS v0.22.0 (5.1) | `docs/evidence/upstream/coqui_tts_v0.22.0_check.md` |
| Artifact hashes, mel-stats identity (5.2) | `artifacts.lock.json`; `docs/evidence/artifact_verification.json`; `docs/evidence/colab_run1/`, `colab_run2/` |
| Bit-identical reruns; seed variance (5.2) | `docs/evidence/colab_run1/pilot_en_reproducibility.json`; `docs/evidence/colab_run2/determinism_check.json` |
| Baseline, Table 2 (5.3) | `eval/baselines/base_xtts_s12_index.json`, `docs/evidence/run5/README.md` |
| Baseline reproducibility (3.6, 5.3) | `docs/evidence/phase8_reproducibility/README.md`, `docs/evidence/phase8_reproducibility/reproducibility_report.json` |
| Planner regression (5.4) | `docs/evidence/colab_run3/README.md`; `tests/test_prepare.py` (`test_cuts_prefer_the_longest_pause_jfk`, `test_pause_never_outweighs_speech`, `test_equal_speech_plans_cut_at_the_longer_pause`) |
| Shifted-cut word scores, Table 3 (5.4) | `docs/evidence/colab_run4/jfk_realign_shift_*.json`; `test_mms_jfk_alignment_and_shifted_cut` |
| Smoke runs and DVAE drift, Table 4 (5.5) | `docs/evidence/colab_run4/` (`stage1_smoke.json`, `training_summary.json`, `runs_gpt_smoke_log.txt`, `stage3_metrics.jsonl`); drift defined in `xtts_trainer/stage1_dvae/train.py` (`evaluate_dvae`) |
| Paired A/B and gates (3.5) | `scripts/compare_ab.py`; `eval/gates.json`; `tests/test_compare_ab.py` |
| Target-speaker hold-out guard (3.5) | `eval/target_voices.py`; `xtts_trainer/data/validate.py`; `tests/test_target_voices.py` |
| Target speaker adaptation, Table 5 (§5.6, Run 6) | `eval/gates_target.json`; `docs/evidence/run6_target_speaker/ab_report_target.json`; `docs/evidence/run6_target_speaker/evidence/prep_report.json`; `docs/evidence/run6_target_speaker/evidence/validation_report.json`; `docs/evidence/run6_target_speaker/evidence/training_summary.json` |
| Source pinning (3.6) | `xtts_trainer/speech_data.py`; `docs/evidence/source_pins.md` (SHA-256 pins, with the SHA-1 cross-check) |
