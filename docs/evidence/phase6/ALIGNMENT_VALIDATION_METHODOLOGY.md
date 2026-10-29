# Multilingual & Multi-Speaker Alignment Threshold Calibration Methodology

Empirically grounds **`docs/recipe_design.md` §2 (Finding F5)** and **§6 (Phase 6 Data Preparation)** as an in-sample threshold calibration (`[E5]`). Held-out validation on unseen voices and languages has since run under a pre-registered protocol (`../phase6_heldout/README.md`): clean acceptance is validated out-of-sample for English (39/40 utterances, 59/60 segments); French and German had too few clips to validate; detection is reliable only for large corruptions.

---

## 1. Experimental Scope & Precision of Claims

- **Scope**: In-sample threshold calibration of MMS-FA forced alignment quality checks across **8 baseline speakers**, 2 languages (English, Castilian Spanish), and 5 accent profiles:
  1. `jfk_english_male` (US English male, 1960s vintage recording)
  2. `arctic_slt_english_female` (US English female, studio clean gold standard)
  3. `arctic_ksp_english_indian_accent` (Indian English male, accented, studio clean)
  4. `arctic_bdl_english_male` (US English male, studio clean)
  5. `arctic_clb_english_female` (US English female, studio clean)
  6. `arctic_awb_english_scottish` (Scottish English male, accented, studio clean)
  7. `ljspeech_female` (US English female, benchmark standard)
  8. `spanish_cuentos_male` (Castilian Spanish male, LibriVox audiobook)

- **Controlled Corruption Matrix**: A 96-test evaluation protocol (8 clean baselines + 88 controlled corruptions, 11 mutations per speaker) across 3 realistic failure modes:
  1. **Edge Truncation Severity Ladder**: 100 ms, 250 ms, 500 ms, 1,000 ms at closing word tail and 500 ms, 1,000 ms at opening word head.
  2. **Hallucinated / Unspoken Word in Transcript**: Token insertions across Head, Mid, and Tail positions (short function words vs. long content words).
  3. **Unexplained Speech Spillover**: Sub-threshold ($+0.10\text{s}$ appended, measuring 0.00 s of unexplained speech) vs. cross-threshold ($+0.50\text{s}$ appended, measuring 0.08–0.14 s) audio from adjacent sentence onsets. The labels describe the audio appended; what the validator compares with the limit is the measured unexplained speech.

- **Defensible Framing (`[E5]` Calibration)**: The benchmark **calibrates thresholds in-sample and characterizes detection limits and specific failure modes**. Setting `max_weak_words = 3` was required to accommodate non-native accented speech (`arctic_ksp`, lowest score 0.062 with 3 weak words), while `max_unexplained_speech_s = 0.10s` was retained since all clean recordings measure 0.00 s. The run itself used 0.20 s; verdicts at 0.10 s are recomputed from the recorded unexplained speech (Section 6). Held-out results: `../phase6_heldout/README.md`.

---

## 2. Validator Threshold Arithmetic & Exact Comparison Operators

The alignment validator evaluates segment quality through `xtts_trainer/prepare/segment.py:segment_problems`. A code audit confirms the exact comparison operators implemented, including boundary equality semantics:

```python
# xtts_trainer/prepare/segment.py:153-172
def segment_problems(segment: Segment, unexplained_speech_s: float, config: Optional[QualityConfig] = None) -> List[str]:
    config = config or QualityConfig()
    problems = []
    worst = min(segment.words, key=lambda w: w.score)
    if worst.score < config.min_word_score:
        problems.append(f"low_alignment_score: '{worst.text}' {worst.score:.2f} < {config.min_word_score}")
    weak = [w for w in segment.words if w.score < config.weak_word_score]
    if len(weak) > config.max_weak_words:
        listed = ", ".join(f"'{w.text}' {w.score:.2f}" for w in weak[:4])
        problems.append(f"weak_alignment: {len(weak)} words below {config.weak_word_score} ({listed}); "
                        f"at most {config.max_weak_words} allowed")
    if unexplained_speech_s > config.max_unexplained_speech_s:
        problems.append(f"unexplained_speech: {unexplained_speech_s:.2f}s of speech not in the transcript")
    for w in segment.words:
        limit = max(config.max_word_s, config.max_s_per_char * len(w.text))
        if w.end_s - w.start_s > limit:
            problems.append(f"stretched_word: '{w.text}' spans {w.end_s - w.start_s:.2f}s")
            break
    return problems
```

### Threshold Summary Table

| Parameter | Config Value | Exact Code Comparison | Behavior at Equality | Rationale |
|---|---|---|---|---|
| `min_word_score` | `0.05` | `worst.score < 0.05` | Score of $0.050$ passes | Severe failure floor. Rejects missing/severely mutilated words. |
| `weak_word_score` | `0.20` | `w.score < 0.20` | Score of $0.200$ is not weak | Classifies unstressed/accented words into the weak-word tolerance band. |
| `max_weak_words` | `3` | `len(weak) > 3` | Exactly 3 weak words passes | Permits natural reduction in accented speech (e.g. Indian English `arctic_ksp`, 3 weak words). 4+ rejects. |
| `max_unexplained_speech_s` | `0.10s` | `unexplained_speech_s > 0.10` | Measured $0.100\text{s}$ passes | Speech detected by CTC emissions outside aligned path. Retained at 0.10s: clean audio measures 0.00s, and 0.10s catches trailing speech (Spanish 0.14s). |
| `max_word_s` | `1.50s` | `(w.end_s - w.start_s) > limit` | Duration equal to limit passes | Prevents unaligned speech from being silently stretched over adjacent words. |
| `max_s_per_char` | `0.35s` | `limit = max(1.5, 0.35 * len(text))` | Proportional to word length | Accommodates drawn-out polysyllabic words while capping pathological stretching. |

### Word Score Calculation Formula
In `xtts_trainer/prepare/align.py:align_emission`, word-level alignment scores are computed from character token posterior probabilities:
$$\text{Score}(W) = \frac{\sum_{s \in \text{spans}} \text{Prob}(s) \times (s_{\text{end}} - s_{\text{start}})}{\sum_{s \in \text{spans}} (s_{\text{end}} - s_{\text{start}})}$$
where $\text{Prob}(s) = \exp(\text{log\_score})$ from `torchaudio.functional.forced_align`.
The score is a **frame-weighted average posterior probability** across the character tokens comprising the word, not a naive unweighted arithmetic mean.

---

## 3. Empirical Results Across 8 Vetted Baseline Speakers

Evaluated on an **NVIDIA A100-SXM4-40GB GPU** via PyTorch 2.1.2 + CUDA 12.1. Full results saved to `docs/evidence/phase6/multilingual_alignment_validation.json`.

### Clean Baseline Verification (Zero False Positives)
- **Distinct Vetted Speakers Tested**: 8 (`jfk`, `arctic_slt`, `arctic_ksp`, `arctic_bdl`, `arctic_clb`, `arctic_awb`, `ljspeech`, `spanish_cuentos_male`).
- **Clean Acceptance Rate**: **8/8 (100%)**.
- Every clean baseline recording across all 8 speakers and 2 languages was accepted without false-positive rejections.
- Accented speech verification: `arctic_ksp` (Indian English male) exhibited 3 weak words with a minimum word score of $0.062$ (and mean word score $0.4879$). Because $0.062 \ge 0.05$ (Tier 1 floor satisfied) and $3 \le 3$ (Tier 2 weak-word limit satisfied), the utterance was correctly accepted.

---

## 4. Observed Rejection Rates by Mutation Type and Severity

Rejection behavior differed substantially between mutation types and boundary locations. Rather than exhibiting a single uniform sensitivity curve, detection rates depend on cut location, acoustic context, and word length:

```
Observed Rejection Rates by Mutation Type and Severity:
---------------------------------------------------------------------
100 ms tail cut:     3/8 rejected  (37.5%)   [5 passed]
250 ms tail cut:     5/8 rejected  (62.5%)   [3 passed]
500 ms tail cut:     2/8 rejected  (25.0%)   [6 passed]
1,000 ms tail cut:   7/8 rejected  (87.5%)   [1 passed]
500 ms head cut:     7/8 rejected  (87.5%)   [1 passed]
1,000 ms head cut:   8/8 rejected  (100.0%)  [0 passed]
---------------------------------------------------------------------
Total Edge Truncation: 32/48 rejected (66.7%)
```

### Analysis of Head vs. Tail Cut Asymmetry
1. **Head Truncation**:
   - 1,000 ms head cut: 8/8 (100%) rejected.
   - 500 ms head cut: 7/8 (87.5%) rejected.
   - Cutting the onset of an utterance immediately amputates the opening word's consonants or primary vowel. The trellis cannot match the opening characters to the missing acoustics, dropping the word score to $0.00 \text{--} 0.04$ ($< 0.05$ hard floor).
2. **Tail Truncation**:
   - 100 ms tail cut: 3/8 (37.5%) rejected.
   - 250 ms tail cut: 5/8 (62.5%) rejected.
   - 500 ms tail cut: 2/8 (25.0%) rejected.
   - 1,000 ms tail cut: 7/8 (87.5%) rejected.
   - Milder tail cuts of 100–250 ms frequently clip trailing silence, post-vocalic aspiration, or nasal release without reducing the core vowel nucleus below the $0.05$ score threshold, producing scores in the $0.08 \text{--} 0.16$ range that remain within the weak-word tolerance band.
   - At 500 ms, detection drops non-monotonically to 25.0% (2/8), lower than the 250 ms rate (62.5%): while a 250 ms cut often clips directly into the nucleus of the final word (dropping its score to 0.01–0.04 and failing the 0.05 floor), a deeper 500 ms cut often amputates the final token entirely or lands in an inter-word boundary, allowing the CTC trellis to realign earlier tokens across the remaining speech frames without triggering the 0.05 hard floor. The two rejected 500 ms cases (`arctic_slt` and `spanish_cuentos_male`) both scored 0.01 and failed the 0.05 floor.
   - At $1,000\text{ ms}$, tail truncation reliably eliminates multiple words or shifts prior syllable boundaries, driving detection to 87.5% (7/8).

---

## 5. Investigation of Failure Mode 1: Transcript Insertions (17/24 Rejected)

Across 24 insertion trials (3 per speaker: head function word, mid content word, tail content word), 17 were rejected and 7 were accepted. A frame-level diagnostic probe (`scripts/diagnose_alignment_edge_cases.py`) on GPU established the exact mechanism for all 7 accepted cases:

| Speaker | Mutation Position | Inserted Token | Inserted Word Score | Utterance Weak Word Count | Acceptance Mechanism |
|---|---|---|---|---|---|
| `jfk_english_male` | Tail Content | *"consequently"* | $0.0869$ | 1 ($\le 3$) | Score above 0.05 floor; absorbed by weak-word tolerance |
| `arctic_slt_english_female` | Tail Content | *"consequently"* | $0.0690$ | 2 ($\le 3$) | Score above 0.05 floor; absorbed by weak-word tolerance |
| `spanish_cuentos_male` | Tail Content | *"consecuentemente"* | $0.0599$ | 1 ($\le 3$) | Score above 0.05 floor; absorbed by weak-word tolerance |
| `ljspeech_female` | Tail Content | *"consequently"* | $0.1666$ | 1 ($\le 3$) | Score above 0.05 floor; absorbed by weak-word tolerance |
| `ljspeech_female` | Head Function | *"Moreover"* | $0.1250$ | 1 ($\le 3$) | Score above 0.05 floor; absorbed by weak-word tolerance |
| `arctic_bdl_english_male` | Mid Content | *"extraordinary"* | **$0.2371$** | 0 ($\le 3$) | Phonetic substring match (/dʒər/ and /tr/); scored above 0.20 weak threshold |
| `arctic_awb_english_scottish` | Mid Content | *"extraordinary"* | **$0.2335$** | 0 ($\le 3$) | Phonetic substring match (/dʒər/ and /tr/); scored above 0.20 weak threshold |

### Diagnostic Findings
1. **Weak-Word Tolerance Absorption (5 cases)**:
   - For trailing words aligned over low-amplitude background noise or terminal silence, MMS-FA emissions average $0.06 \text{--} 0.16$ across the character sequence.
   - Because $0.0599 \text{--} 0.1666 \ge 0.05$, the hard floor does not trigger.
   - Because only 1 or 2 words in the utterance scored below $0.20$, `len(weak) <= 3`, and the segment passes.
2. **Acoustic Substring Overlap (2 cases)**:
   - In `arctic_bdl` and `arctic_awb`, inserting *"extraordinary"* between *"danger"* and *"trail"* caused the aligner to map tokens to acoustically similar phones in adjacent words (/dʒər/ and /tr/).
   - The inserted word scored $> 0.20$, so it was not even classified as a weak word.

---

## 6. Investigation of Failure Mode 2: Speech Spillover (+500 ms Appended Audio)

In the spillover benchmark:
- **Sub-threshold condition ($+0.10\text{s}$ appended)**: 8/8 correctly accepted (tolerance functioning as intended).
- **Cross-threshold condition ($+0.50\text{s}$ appended)**: 8/8 passed, evading the expected rejection.

These counts were measured with `max_unexplained_speech_s = 0.20`, the setting at run time (`multilingual_alignment_validation.json`, `quality_config`). At the operating 0.10 s limit, recomputed from the recorded `unexplained_speech_s` (no rerun; only this rule depends on the limit, and no other mutation measured between 0.10 and 0.20 s): sub-threshold 8/8 passed, cross-threshold 7/8 passed (Spanish, 0.14 s, rejected), so 58/88 corruptions are classified correctly instead of 57/88. The sub-threshold case is not on the 0.10 s boundary: the 0.10 s appended fell inside the inter-utterance pause, and every speaker measured 0.00 s of unexplained speech.

### Frame-Level Diagnostic on Representative Cases
To understand why $+500\text{ms}$ appended audio was not flagged, frame-level trellis paths, token spans, and RMS speech energies were measured on 3 representative speakers on GPU:

| Metric | `jfk_english_male` | `arctic_slt_english_female` | `spanish_cuentos_male` |
|---|---|---|---|
| Inter-utterance silence before appended speech | $0.340\text{s}$ | $0.280\text{s}$ | $0.310\text{s}$ |
| Appended slice RMS energy | $0.1843$ (speech) | $0.1359$ (speech) | $0.0539$ (speech) |
| Terminal word duration in clean clip | $0.360\text{s}$ (*"ask"*) | $0.300\text{s}$ (*"etc."*) | $0.620\text{s}$ (*"años"*) |
| Terminal word duration with $+500\text{ms}$ | $0.379\text{s}$ | $0.280\text{s}$ | $0.620\text{s}$ |
| Terminal word duration growth | $+0.019\text{s}$ | $-0.020\text{s}$ | $0.000\text{s}$ |
| Trellis assignment of trailing frames | **BLANK** tokens | **BLANK** tokens | **BLANK** tokens |
| MMS-FA argmax heard non-blank frames | 6 frames | 6 frames | 11 frames |
| **Measured `unexplained_speech_s`** | **$0.0801\text{s}$** | **$0.0802\text{s}$** | **$0.1401\text{s}$** |
| Threshold (`max_unexplained_speech_s`) | $0.1000\text{s}$ | $0.1000\text{s}$ | $0.1000\text{s}$ |
| Validator Verdict under 0.10s | Passed ($0.0801 \le 0.10$) | Passed ($0.0802 \le 0.10$) | **REJECTED** ($0.1401 > 0.10$) |

> [!NOTE]
> Under the reverted 0.10 s threshold, the Spanish trailing speech intrusion (measuring 140 ms of active speech) is correctly flagged and rejected. In JFK and SLT, the initial 280–340 ms consisted of silence, leaving only 80 ms of active speech in the appended slice, which passed below the 100 ms threshold. Relaxing the threshold to 0.20 s had created an unnecessary blind spot that let Spanish escape.

### Key Technical Mechanism
1. **No Terminal Word Stretching**:
   The CTC trellis does not force the terminal transcript word to stretch across the appended speech. Across all three cases, the terminal word boundary shifted by at most $19\text{ ms}$. Trailing frames were assigned to the CTC blank token (`-`).
2. **Measurement vs. Injected Duration**:
   While $500\text{ ms}$ of audio was appended, the initial $280 \text{--} 340\text{ ms}$ consisted of inter-utterance pause/silence. Active speech onset occurred only in the final $160 \text{--} 220\text{ ms}$.
3. **Argmax Emission Sensitivity**:
   MMS-FA's argmax emissions detected non-blank speech characters during these trailing frames. Because clean speech measures $0.00\text{ s}$ unexplained speech, retaining `max_unexplained_speech_s = 0.10s` maintains clean acceptance while catching trailing intrusions exceeding 100 ms.

---

## 7. Conclusions & Next Steps

1. **In-Sample Calibration Status (`[E5]`)**: The 8-speaker benchmark characterizes threshold behavior in-sample. Setting `max_weak_words = 3` accommodates non-native accented speech (`arctic_ksp`, 3 weak words scoring down to 0.062), but narrows the margin above the 0.05 floor.
2. **Boundary Sensitivity**: Detection of edge truncation is non-monotonic: 37.5% at 100 ms, 62.5% at 250 ms, 25.0% at 500 ms (where whole-word loss allows trellis realignment without dropping below 0.05), and 87.5% at 1,000 ms. Head truncation is detected at 87.5%–100%.
3. **Unexplained Speech Retained at 0.10 s**: All clean recordings measure 0.00 s, and 0.10 s catches trailing speech intrusions (Spanish 0.14 s). The run used 0.20 s; the 0.10 s verdicts are recomputed from the recorded values, not rerun.
4. **Held-Out Validation (done; `../phase6_heldout/README.md`)**: Run under a pre-registered protocol in the locked environment, on new CMU ARCTIC voices (`rms`, `jmk`) and LibriVox readers in French and German, with full per-word scores logged. Clean acceptance is validated out-of-sample for English (39/40 utterances, 59/60 segments); French and German had too few clips to validate; only large corruptions are reliably detected.
