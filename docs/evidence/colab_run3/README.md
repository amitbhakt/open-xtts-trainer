# Colab run 3: Phases 5–7 with real weights (2026-10-24)

This record is transcribed from the notebook output pasted into the session. No evidence zip came back,
because the run stopped at step 7c.

## Passed

| Step | Test | Result |
|---|---|---|
| 7 | `test_vocoder_round_trip_prefers_true_alignment` | PASSED: teacher-forced latents through the stock decoder match the original best at the true alignment |
| 7 | `test_upstream_checkpoint_loads_into_factory_models` | PASSED |
| 7 | `test_upstream_dvae_loads` (F3 allowlist on the real `dvae.pth`) | PASSED |
| 7 | `test_real_vocab_unk_detection` | PASSED |
| 7b | `test_mms_jfk_alignment_and_shifted_cut[0.0]` | PASSED: JFK clips cut from MMS_FA alignment re-align cleanly |
| 7b | `test_mms_jfk_alignment_and_shifted_cut[+1.0]`, `[-1.0]` | PASSED: a cut moved 1 s either way fails validation (`alignment_mismatch`); this meets the Phase 6 exit criterion on real audio |

MMS_FA weights: 1.18 GB, downloaded in 5 s.

## Failed, and fixed locally

**Step 7b, sample set:**
- LJSpeech (6 clips) prepared fine. JFK yielded one clip: "And so, my fellow Americans: ask not what your
  country can do" (0–6.88 s). The second clip, "for you, ask what you can do for your country." (6.88–10.62 s),
  was dropped for `low_alignment_score: 'for' 0.12 < 0.2`.
- With one JFK clip, validation refused the dataset (`single_clip_speaker`).
- Cause: the cut planner scored the audio it kept, so cutting at a long pause looked worse, because the pause
  itself is dropped. It cut at the tight "do | for" boundary instead of the 1.06 s pause after "not", which
  the run's own warning reported as `unpunctuated_pause after 'not' gap 1.062s`.
- Fix: plans now score kept *speech* plus a preference for longer pauses (`plan_segments`). The JFK-shaped
  regression test (`test_cuts_prefer_the_longest_pause_jfk`) reproduces the old "do" cut under the old
  objective and passes with the new one.

**Step 7c, Stage 2:** no dataset, so `train_gpt` failed and the cell crashed on the missing summary; its
`grep` filter hid the error. The cell now keeps the full log, prints its tail and stops with a clear message.

## Still to observe (next run)

- the sample set prepared and validated with zero edits, with JFK in two or three clips;
- Stage 2 eval loss before and after on real weights, strict reload of the export, synthesis.
