# Training and Prediction

Run commands from the repository root. Paths in configs are relative to the
working directory. Inputs and all generated files stay local.

```bash
uv run python scripts/train/run_training.py --config configs/paper/gru_fitbit_only_60m.yaml
uv run python scripts/train/run_mortality_cv.py --config configs/paper/mortality_cv_primary_60m.yaml
```

For a single model, generate the matching single-split directory first (see
[data preparation](data.md)). For CV, generate the fold root from
`configs/data/splits.yaml`. The CV runner overrides each model's split directory
with the current fold. It writes under `study.output_dir`, defaulting to
`outputs/<study_id>`. `study.model_ids` can select a subset and `study.overrides`
accepts a list of dot-path assignments applied to each model config. CV runs
sequentially, with seed `training_seed + fold_index`.

## Data and Model Settings

`data.parquet_path` contains already-derived binary horizon labels and activity
epochs. `data.participants_path`, when supplied, joins participant metadata to
predictions. Sequence fields are intensity, hour (0-23), weekday (1-7), and a
binary attention flag. Windows are sorted by `PAXN` when available; otherwise
configure `windowing.time_columns` explicitly for your data.

`split.mode` is `runtime` (stratified splitting) or `precomputed` (three ID files).
Partitions must be disjoint and cover the prepared cohort. Window filtering can
remove participants; assigned/retained counts are recorded for every partition,
and both outcome classes must remain.

Intensity normalization uses observed training-window values only. Covariate
imputation, scaling and category vocabularies are also fitted only on training
participants with retained windows. Choose fields using `data.covariates` or its
`metadata_path`. Numeric input dimensions and category embedding cardinalities
are derived from those fields and the training vocabulary, overriding template
dimensions. Unknown categories in other partitions use the reserved zero code.
The exact resolved architecture and fitted preprocessing are saved with the model.

All GRU, LSTM and Transformer binary variants use the existing model factory.
The workflow uses AdamW, optional parameter groups, gradient clipping and a
validation-driven plateau scheduler. Warm-starting uses `experiment.init_checkpoint`.
Only load checkpoints from sources you trust. This runner does not implement
historical tuning searches, staged freezing or training-state resume.

## Selection and Refitting

By default `task.selection_metric` selects the best checkpoint using participant
validation probabilities (the mean of per-window sigmoid probabilities).
Early stopping and scheduling use the same metric. The binary decision threshold
is chosen on validation data, never test data. Evaluation uses every retained
window; it is not truncated by the legacy `limit_val_batches` helper setting.
`training.limit_train_batches` is available for small smoke runs.

Set `training.fixed_epochs` for a predefined epoch budget without validation
selection. To refit on combined train/validation participants, set:

```yaml
final_refit:
  enabled: true
  epochs: 20
  combine_train_val: true
```

The epoch budget must be chosen without looking at the outer test results.
Refitting learns preprocessing on the combined development set and uses a fixed
0.5 decision threshold. Fixed-epoch runs do not use the validation scheduler.

## Outputs and Reuse

Each run writes `model.pt`, `resolved_config.yaml`, `preprocessing.json`,
`participant_predictions.parquet`, `training_log.json`, and aggregate
`summary.json` under `experiment.output_dir`. Existing nonempty run directories
are rejected. CV adds a manifest, per-fold metrics and generic mean/SD summaries;
these outputs have no publication-specific table layout.

Apply a saved model without refitting preprocessing using a prediction config:

```yaml
checkpoint_dir: outputs/my_model
data:
  parquet_path: data/processed/new_activity.parquet
  participants_path: data/processed/new_participants.parquet
output_path: outputs/new_predictions.parquet
device: cpu
```

```bash
uv run python scripts/predict/predict.py --config configs/examples/prediction.yaml
```

The checked-in prediction config reuses the default synthetic workflow's model
and inputs. Adapt its paths as illustrated above for new participants. Covariate models also
need `data.covariates.parquet_path` pointing to the new participant covariates.
Mortality labels are optional at prediction time. Preprocessing state and model
weights are loaded from the training directory and are never refitted.
