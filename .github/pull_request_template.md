<!-- docs/recipe_design.md §7: the bug-class protocol. Fill in every section, including for your own changes. -->

## What changed, and the class it belongs to
<!-- The class ("spectrogram settings differ from upstream"), not only the instance ("hop is 1024 in dataset.py"). -->

## Failing test first
<!-- Link the test. Paste how it failed on the old code. -->

## Sibling search
<!-- Paste the grep/rg commands and what they found; say which siblings were fixed. -->

## Source of truth
<!-- Which single module owns this fact now (xtts_trainer/spec, frontend, models, data, ...)? -->

## External facts and their evidence
<!-- Every number or text from outside the repo (hash, timestamp, transcript, language support) with the command
output or file that establishes it. Nothing typed in. -->

## Mutation check
<!-- Break the fix on purpose: which test fails? A test that can't fail proves nothing. -->

## Measurements
<!-- If model behaviour can change: harness manifests before/after and scripts/compare_ab.py output. Otherwise "n/a". -->

## Checklist
- [ ] One concern in this PR; commit messages say what was verified, not what was "resolved".
- [ ] No test was edited to match the code; tests change only when a requirement changes.
- [ ] No new `try/except` fallback, `strict=False`, default language/speaker, or random data in a data path.
- [ ] Thresholds come from the baseline, a reference or a stated requirement, with the reason next to the number.
