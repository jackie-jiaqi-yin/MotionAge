from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from motionage.data.partitioning import build_partitions, make_splits
from motionage.preprocessing.pipeline import activity_blocks, prepare_activity, prepare_inputs
from motionage.workflow_io import read_table


def raw_activity(n=4, minutes=20):
    return pd.DataFrame({
        "SEQN": np.repeat(np.arange(1, n + 1), minutes), "PAXN": np.tile(np.arange(1, minutes + 1), n),
        "PAXDAY": 1, "PAXHOUR": 0, "PAXMINUT": np.tile(np.arange(minutes) % 60, n),
        "PAXINTEN": 150.0, "PAXSTAT": 1, "PAXCAL": 1,
    })


def test_chunk_boundaries_and_preparation(tmp_path):
    activity = raw_activity()
    activity.to_csv(tmp_path / "activity.csv", index=False)
    blocks = list(activity_blocks(tmp_path / "activity.csv", 13))
    assert sum(len(block) for block in blocks) == len(activity)
    assert sum(block.SEQN.nunique() for block in blocks) == 4
    participants = pd.DataFrame({"SEQN": range(1, 5), "RIDAGEYR": [40, 50, 60, 70], "RIAGENDR": [1, 2, 1, 2]})
    participants.to_csv(tmp_path / "participants.csv", index=False)
    pd.DataFrame({"SEQN": range(1, 5), "mortstat": [0, 1, 0, 1], "permth_int": [120, 40, 20, 80]}).to_csv(tmp_path / "mortality.csv", index=False)
    config = {
        "inputs": {"activity": [str(tmp_path / "activity.csv")], "participants": [{"files": [str(tmp_path / "participants.csv")]}], "mortality": [str(tmp_path / "mortality.csv")]},
        "output_dir": str(tmp_path / "processed"), "chunksize": 13,
        "covariate_bundles": {"age": {"numeric_columns": ["RIDAGEYR"], "categorical_columns": ["RIAGENDR"]}},
    }
    summary = prepare_inputs(config)
    assert summary["participants"] == 3
    assert summary["events"] == 1
    prepared = pd.read_parquet(tmp_path / "processed/activity_mortstat/activity_mortstat_joined.parquet")
    assert len(prepared) == 12
    assert prepared.intensity_mean.eq(150).all()
    assert set(prepared.SEQN) == {1, 2, 4}
    with pytest.raises(ValueError, match="already exists"):
        prepare_inputs(config)


def test_minute_gaps_do_not_create_wear_or_nonwear():
    frame = raw_activity(n=1).drop(index=range(5, 10))
    result = prepare_activity(frame, {"epoch_minutes": 5})
    assert result.attention_flag.tolist() == [1, 0, 1, 1]
    assert result.intensity_mean.tolist() == [150, 0, 150, 150]
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_activity(pd.concat([frame, frame.iloc[:1]]), {})


def test_stratified_cv_is_deterministic_disjoint_and_complete(tmp_path):
    participants = pd.DataFrame({"SEQN": np.arange(100), "mortstat": np.arange(100) % 2})
    config = {"n_folds": 5, "random_seed": 9, "val_size": 0.25}
    folds = build_partitions(participants, config)
    shuffled = build_partitions(participants.sample(frac=1), config)
    all_tests = []
    for name, split in folds.items():
        assert not set(split["train"]) & set(split["test"])
        assert not set(split["val"]) & set(split["test"])
        assert not set(split["train"]) & set(split["val"])
        assert sum(map(len, split.values())) == 100
        for part in split:
            np.testing.assert_array_equal(split[part], shuffled[name][part])
        all_tests.extend(split["test"])
    assert sorted(all_tests) == list(range(100))
    participants.to_parquet(tmp_path / "participants.parquet")
    summary = make_splits({**config, "participants_path": tmp_path / "participants.parquet", "output_dir": tmp_path / "splits"})
    assert summary["n_folds"] == 5
    assert (tmp_path / "splits/fold_0/val_ids.csv").exists()
    with pytest.raises(ValueError, match="duplicate"):
        build_partitions(pd.concat([participants, participants.iloc[:1]]), config)


def test_mortality_fixed_width_reader(tmp_path):
    row = list(" " * 61)
    row[0:6] = " 12345"
    row[14:16] = "11"
    row[42:45] = " 48"
    row[45:48] = " 47"
    path = tmp_path / "mortality.dat"
    path.write_text("".join(row) + "\n")
    frame = read_table(path)
    assert frame.iloc[0].to_dict() == {"SEQN": 12345, "eligstat": 1, "mortstat": 1, "permth_int": 48, "permth_exm": 47}


def test_unsorted_activity_is_rejected(tmp_path: Path):
    path = tmp_path / "activity.csv"
    raw_activity().iloc[::-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="sorted"):
        list(activity_blocks(path, 100))
