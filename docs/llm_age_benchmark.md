# LLM-Age Benchmark

The `motionage.benchmarks.llm_age` module evaluates an optional external
LLM-age prediction table against 60-month mortality outcomes. The repository
does not include collaborator prediction CSVs, prompt outputs, participant-level
processed tables, or fold split IDs.

Expected local inputs:

- an external CSV with `SEQN`, `mortstat`, `permth_int`, `RIDAGEYR`,
  `RIAGENDR`, and `overall_age`;
- optional `llm_ok` quality flags;
- local fold directories containing `train_ids.csv` and `test_ids.csv`.

The loader converts mortality to the paper's 60-month label, renames
`overall_age` to `LLMOverallAge`, and computes `LLMOverallAgeAccel` as
`LLMOverallAge - RIDAGEYR`. Fold benchmarks fit logistic models for three
feature sets: chronological age plus sex, raw LLM-age plus sex, and LLM-age
acceleration plus chronological age plus sex.

Generated fold outputs should be written under ignored local output directories,
not committed to git.
