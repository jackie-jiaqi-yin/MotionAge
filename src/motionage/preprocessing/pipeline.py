"""Configuration-driven preparation of local NHANES inputs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from motionage.preprocessing.mortality import build_fixed_horizon_mortality_table
from motionage.preprocessing.wear import detect_nonwear_choi, downsample_wear_epochs
from motionage.workflow_io import read_table, require_columns, validate_ids, write_json


def _activity_chunks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if path.suffix.lower() in {".xpt", ".zip"}:
        yield from pd.read_sas(path, format="xport", encoding="utf-8", chunksize=chunksize)
    elif path.suffix.lower() == ".csv":
        yield from pd.read_csv(path, chunksize=chunksize)
    elif path.suffix.lower() == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=chunksize):
            yield batch.to_pandas()
    else:
        raise ValueError("Activity input must be CSV, Parquet, XPT, or a single-XPT ZIP.")


def activity_blocks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    """Read sorted inputs without splitting a participant across processing blocks."""
    carry = pd.DataFrame()
    last_id = None
    for chunk in _activity_chunks(path, chunksize):
        chunk = validate_ids(chunk, "SEQN", unique=False)
        if chunk.empty:
            continue
        if not chunk.SEQN.is_monotonic_increasing or (last_id is not None and chunk.SEQN.iloc[0] < last_id):
            raise ValueError("Activity files must be sorted by SEQN for chunked processing.")
        last_id = chunk.SEQN.iloc[-1]
        combined = pd.concat([carry, chunk], ignore_index=True)
        mask = combined.SEQN == last_id
        carry = combined.loc[mask].copy()
        if (~mask).any():
            yield combined.loc[~mask]
    if not carry.empty:
        yield carry


def prepare_activity(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Preserve minute gaps as masked observations before epoch aggregation."""
    columns = ["SEQN", "PAXN", "PAXDAY", "PAXHOUR", "PAXMINUT", "PAXINTEN", "PAXSTAT", "PAXCAL"]
    require_columns(frame, columns)
    frame = validate_ids(frame[columns], "SEQN", unique=False)
    pieces = []
    for _, participant in frame.groupby("SEQN", sort=True):
        participant = participant.sort_values("PAXN").copy()
        minute = pd.to_numeric(participant.PAXN, errors="raise")
        if minute.isna().any() or (minute % 1 != 0).any() or not minute.between(1, 10080).all():
            raise ValueError("PAXN must contain integer minute indices between 1 and 10080.")
        if minute.duplicated().any():
            raise ValueError("Duplicate participant/minute activity records.")
        participant["PAXN"] = minute.astype(int)
        first = participant.iloc[0]
        for column, upper in [("PAXDAY", 7), ("PAXHOUR", 23), ("PAXMINUT", 59)]:
            values = pd.to_numeric(participant[column], errors="raise")
            lower = 1 if column == "PAXDAY" else 0
            if not values.between(lower, upper).all() or (values % 1 != 0).any():
                raise ValueError(f"Invalid activity time column: {column}")
        participant = participant.set_index("PAXN").reindex(range(1, int(minute.max()) + 1))
        participant.index.name = "PAXN"
        participant["SEQN"] = int(first.SEQN)
        elapsed = participant.index.to_numpy() - int(first.PAXN)
        clock = int(first.PAXHOUR) * 60 + int(first.PAXMINUT) + elapsed
        participant["PAXDAY"] = (int(first.PAXDAY) - 1 + clock // 1440) % 7 + 1
        participant["PAXHOUR"] = (clock // 60) % 24
        participant["PAXMINUT"] = clock % 60
        intensity = pd.to_numeric(participant.PAXINTEN, errors="coerce")
        valid = participant.PAXSTAT.eq(1) & intensity.between(0, 32767)
        if config.get("require_calibration", True):
            valid &= participant.PAXCAL.eq(1)
        if not valid.any():
            continue
        participant["PAXINTEN"] = intensity.where(valid, 0).mask(intensity.abs() < 1e-10, 0)
        wear = np.zeros(len(participant), dtype=int)
        # Missing/invalid minutes break a run; they cannot extend a non-wear interval.
        indices = np.flatnonzero(valid.to_numpy())
        for segment in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
            if len(segment):
                wear[segment] = ~detect_nonwear_choi(
                    participant.PAXINTEN.iloc[segment].to_numpy(), **config.get("nonwear", {})
                )
        participant["wear_flag"] = wear
        pieces.append(downsample_wear_epochs(
            participant.reset_index(), interval=int(config.get("epoch_minutes", 5)),
            tau=float(config.get("attention_tau", 0.2)),
        ))
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def _participants(config: dict) -> pd.DataFrame:
    cycles = []
    for cycle in config["inputs"]["participants"]:
        merged = None
        for filename in cycle["files"]:
            source = validate_ids(read_table(filename), "SEQN")
            if merged is None:
                merged = source
            else:
                overlapping = (set(merged) & set(source)) - {"SEQN"}
                if overlapping:
                    raise ValueError(f"Participant source columns overlap: {sorted(overlapping)}")
                merged = merged.merge(source, on="SEQN", how="outer", validate="one_to_one")
        if merged is None:
            raise ValueError("Each participant cycle needs at least one input file.")
        cycles.append(merged)
    participants = validate_ids(pd.concat(cycles, ignore_index=True), "SEQN")
    for column, codes in config.get("missing_values", {}).items():
        require_columns(participants, [column])
        participants[column] = participants[column].replace(codes, np.nan)
    for column, sources in config.get("mean_features", {}).items():
        require_columns(participants, sources)
        participants[column] = participants[sources].apply(pd.to_numeric, errors="raise").mean(axis=1)
    require_columns(participants, ["RIDAGEYR", "RIAGENDR"])
    participants = participants.loc[participants.RIDAGEYR >= config.get("min_age", 18)].copy()
    raw_mortality = pd.concat([read_table(path) for path in config["inputs"]["mortality"]], ignore_index=True)
    raw_mortality = raw_mortality.rename(columns={"MORTSTAT": "mortstat", "PERMTH_INT": "permth_int", "ELIGSTAT": "eligstat"})
    raw_mortality = validate_ids(raw_mortality, "SEQN")
    if "eligstat" in raw_mortality:
        raw_mortality = raw_mortality.loc[raw_mortality.eligstat == 1]
    horizon = int(config.get("followup_months", 60))
    if horizon <= 0:
        raise ValueError("followup_months must be positive.")
    require_columns(raw_mortality, ["mortstat", "permth_int"])
    months = pd.to_numeric(raw_mortality.permth_int, errors="raise")
    if (months.dropna() < 0).any():
        raise ValueError("Follow-up months cannot be negative.")
    if config.get("require_complete_followup", True):
        raw_mortality = raw_mortality.loc[months.ge(horizon) | (raw_mortality.mortstat.eq(1) & months.notna())]
    mortality = build_fixed_horizon_mortality_table(
        raw_mortality, id_column="SEQN", mortality_column="mortstat", followup_months=horizon,
    )
    return participants.merge(mortality, on="SEQN", how="inner", validate="one_to_one")


def prepare_inputs(config: dict) -> dict:
    """Write local model-ready activity, participant metadata, and covariate bundles."""
    participants = _participants(config)
    if participants.empty:
        raise ValueError("No participants satisfy the configured cohort and mortality criteria.")
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    activity_path = root / "activity_mortstat" / "activity_mortstat_joined.parquet"
    if activity_path.exists():
        raise ValueError("Prepared activity output already exists; choose a new output_dir.")
    bundles = config.get("covariate_bundles", {})
    for name, bundle in bundles.items():
        if not name.replace("_", "").isalnum():
            raise ValueError("Covariate bundle names must be simple identifiers.")
        require_columns(participants, bundle["numeric_columns"] + bundle["categorical_columns"])
    activity_path.parent.mkdir(parents=True, exist_ok=True)
    pending = activity_path.with_suffix(".parquet.partial")
    writer = None
    retained_ids: set[int] = set()
    seen_ids: set[int] = set()
    rows = 0
    try:
        for path in config["inputs"]["activity"]:
            for block in activity_blocks(Path(path), int(config.get("chunksize", 250000))):
                ids = set(block.SEQN.unique())
                if seen_ids & ids:
                    raise ValueError("Participants occur in more than one activity input block/file.")
                seen_ids.update(ids)
                block = block.loc[block.SEQN.isin(participants.SEQN)]
                if block.empty:
                    continue
                prepared = prepare_activity(block, config.get("preprocessing", {}))
                if prepared.empty:
                    continue
                prepared = prepared.merge(
                    participants[["SEQN", "mortstat", "permth_int"]], on="SEQN", validate="many_to_one",
                )
                table = pa.Table.from_pandas(prepared, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(pending, table.schema)
                writer.write_table(table)
                rows += len(prepared)
                retained_ids.update(prepared.SEQN.unique())
        if writer is None:
            raise ValueError("No valid activity remains after preparation.")
    except BaseException:
        if writer is not None:
            writer.close()
            writer = None
        pending.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()
    pending.replace(activity_path)
    participants = participants.loc[participants.SEQN.isin(retained_ids)].sort_values("SEQN")
    participants.to_parquet(root / "participants.parquet", index=False)
    cov_dir = root / "nhanes_mortality_covariates"
    cov_dir.mkdir(exist_ok=True)
    for name, bundle in bundles.items():
        columns = list(dict.fromkeys(["SEQN", "mortstat", "permth_int", *bundle["numeric_columns"], *bundle["categorical_columns"]]))
        participants[columns].to_parquet(cov_dir / f"nhanes_mortality_covariates_{name}.parquet", index=False)
        metadata = {"id_column": "SEQN", **bundle}
        (cov_dir / f"metadata_{name}.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    summary = {
        "participants": len(participants), "activity_rows": rows,
        "events": int(participants.mortstat.sum()), "followup_months": int(config.get("followup_months", 60)),
        "require_complete_followup": config.get("require_complete_followup", True),
        "preprocessing": config.get("preprocessing", {}),
    }
    write_json(root / "preparation_summary.json", summary)
    return summary
