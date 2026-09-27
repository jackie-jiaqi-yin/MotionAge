# Data

MotionAge uses public NHANES accelerometer, demographic, mortality, and covariate files. The repository documents how to prepare these inputs locally but does not redistribute raw or processed participant-level data.

## Public Sources

Core accelerometer and demographic inputs:

- NHANES 2003-2004 minute-level accelerometer data: `PAXRAW_C`
- NHANES 2005-2006 minute-level accelerometer data: `PAXRAW_D`
- NHANES 2003-2004 demographics: `DEMO_C`
- NHANES 2005-2006 demographics: `DEMO_D`
- NHANES linked mortality follow-up files
- NHANES laboratory, examination, and questionnaire covariates used by the paper feature sets

Download files locally using the [NHANES data portal](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2003)
and the [NCHS linked mortality archive](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/).
The activity schema follows the [PAXRAW codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/PAXRAW_C.htm).
The `.dat` reader uses the NHANES positions in the archive's 2019 SAS read-in program.
No files are downloaded automatically.

## Local Layout

The default paths in `configs/data/nhanes.yaml` use this local layout:

```text
data/
  raw/
    nhanes/
      2003-2004/
      2005-2006/
    mortality/
  processed/
    activity_mortstat/
    nhanes_mortality_covariates/
    participants.parquet
    splits/
```

The entire `data/` directory is ignored by git.

## Privacy and Redistribution

Do not commit participant-level data, exact split ID files, or merged processed tables. Even when source files are public, this repository should avoid redistributing derived participant-level datasets unless an explicit release decision is made.

## Validation Expectations

Preparation scripts should write validation summaries with:

- input file names,
- row counts,
- participant counts,
- date or cycle coverage,
- missingness summaries for covariate feature sets,
- the preprocessing and follow-up settings used.

## Preparation and Splits

Run from the repository root; file paths in configs are relative to the working directory:

```bash
uv run python scripts/data/prepare_nhanes_inputs.py --config configs/data/nhanes.yaml
uv run python scripts/data/make_splits.py --config configs/data/splits.yaml
```

Inputs may be CSV, Parquet, or SAS XPORT (`.xpt`, including a single-XPT ZIP).
Mortality inputs also accept the NCHS 2019 NHANES fixed-width `.dat` format.
Each `inputs.participants` entry joins one cycle's demographic/examination files
one-to-one on `SEQN`; cycles are then concatenated. Duplicate IDs and overlapping
non-ID source columns are rejected rather than silently selecting a value.
Participant metadata must contain `RIDAGEYR` and `RIAGENDR`; mortality inputs
must contain `SEQN`, `mortstat`, and `permth_int`. Raw uppercase mortality names
are accepted. An `eligstat` field, when present, restricts inclusion to value 1.

Activity files must be sorted by `SEQN` and include `PAXN`, `PAXDAY`, `PAXHOUR`,
`PAXMINUT`, `PAXINTEN`, `PAXSTAT`, and `PAXCAL`. Reading is chunked at participant
boundaries. Each participant's minutes are sorted by `PAXN`, not day of week.
Invalid and missing minutes remain masked on the time grid, and break non-wear
detection runs. Epoch intensity averages worn minutes only. The attention flag
uses the configured fraction of worn minutes. All-invalid participants are excluded.

`min_age`, calibration filtering, epoch length, attention threshold and Choi-style
non-wear parameters are explicit config values. `missing_values` lists sentinel
codes to recode before `mean_features` derives features such as averaged blood
pressure. `covariate_bundles` declares numeric and categorical fields without
fitting imputation or scaling; those are fitted later on each training partition.
The default feature bundles are public input examples, not a claim of exact
agreement with any historical experiment's selected columns.

The binary target is death within `followup_months` (default 60). With the default
`require_complete_followup: true`, survivors with insufficient or missing follow-up
are excluded; deaths with a known follow-up time remain eligible. Setting this
option to false uses the original indicator-only labeling convention. This choice
changes the cohort and is recorded in the preparation summary.

Preparation writes activity Parquet, participant metadata, covariate Parquet/YAML,
and an aggregate `preparation_summary.json` under `output_dir`. Existing prepared
activity is not overwritten. Use a new directory for another preprocessing variant.

Splitting is participant-level and outcome-stratified. `n_folds: 1` creates
train/validation/test files in `output_dir`; larger values create outer test folds
under `fold_0`, `fold_1`, etc. `val_size` is the fraction of each outer development
set reserved for validation. Every participant is tested once across outer folds.
All three partitions must contain both classes. The seed and aggregate split
counts are recorded in `manifest.json`; ID files stay local.

Override individual values without editing a preset:

```bash
uv run python scripts/data/make_splits.py --config configs/data/splits.yaml --set n_folds=1 --set output_dir=data/processed/splits/mortstat_60m_seed42
```
