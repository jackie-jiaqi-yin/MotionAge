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

The implementation PRs will add scripts that fetch or validate these files from official NHANES/CDC locations.

## Local Layout

Use this local layout when data preparation scripts are added:

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
- checksums for generated local outputs.
