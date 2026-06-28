from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motionage.analysis.motionage.mapping import (
    LogisticInverseGroupFit,
    MotionAgeMapping,
    apply_motionage_mapping,
    fit_motionage_mapping,
    mapping_to_jsonable,
)


def _participants() -> pd.DataFrame:
    rows = []
    for sex_value, base_probability in [(1, 0.12), (2, 0.10)]:
        for age, probability in [
            (40, base_probability),
            (50, base_probability + 0.10),
            (60, base_probability + 0.22),
        ]:
            rows.append(
                {
                    "SEQN": f"{sex_value}-{age}-a",
                    "split": "train",
                    "RIAGENDR": sex_value,
                    "RIDAGEYR": age,
                    "participant_probability": probability,
                }
            )
            rows.append(
                {
                    "SEQN": f"{sex_value}-{age}-b",
                    "split": "validation",
                    "RIAGENDR": sex_value,
                    "RIDAGEYR": age,
                    "participant_probability": probability + 0.01,
                }
            )
    rows.append(
        {
            "SEQN": "held-out",
            "split": "test",
            "RIAGENDR": 1,
            "RIDAGEYR": 55,
            "participant_probability": 0.30,
        }
    )
    return pd.DataFrame(rows)


def test_fit_motionage_mapping_is_sex_specific_and_train_only() -> None:
    mapping, diagnostics = fit_motionage_mapping(
        _participants(),
        fit_partitions=["train"],
        representative_probability="median",
        clip_eps=1.0e-4,
        weighted_fit=True,
        clamp_output_to_fit_age_range=False,
        require_positive_beta=True,
    )

    assert mapping.method == "sex_specific_logistic_inverse"
    assert mapping.fit_partitions == ("train",)
    assert {group.sex_value for group in mapping.groups} == {1, 2}
    assert all(group.beta > 0 for group in mapping.groups)
    assert diagnostics["split"].isna().all()
    assert set(diagnostics["sex_value"]) == {1, 2}
    assert diagnostics.groupby("sex_value")["n_participants"].sum().to_dict() == {1: 3, 2: 3}


def test_apply_motionage_mapping_outputs_motionage_and_acceleration() -> None:
    mapping, _ = fit_motionage_mapping(
        _participants(),
        fit_partitions=["train"],
        representative_probability="mean",
        clip_eps=1.0e-4,
        weighted_fit=False,
        clamp_output_to_fit_age_range=False,
        require_positive_beta=True,
    )

    mapped = apply_motionage_mapping(_participants(), mapping)

    held_out = mapped.loc[mapped["SEQN"] == "held-out"].iloc[0]
    assert np.isfinite(held_out["MotionAge"])
    assert held_out["MotionAgeAccel"] == pytest.approx(held_out["MotionAge"] - 55)


def test_apply_motionage_mapping_clips_and_clamps_outputs() -> None:
    mapping = MotionAgeMapping(
        method="sex_specific_logistic_inverse",
        fit_partitions=("train",),
        representative_probability="mean",
        clip_eps=0.01,
        weighted_fit=False,
        clamp_output_to_fit_age_range=True,
        require_positive_beta=True,
        age_column="RIDAGEYR",
        sex_column="RIAGENDR",
        probability_column="participant_probability",
        groups=(
            LogisticInverseGroupFit(
                sex_value=1,
                alpha=-8.0,
                beta=0.1,
                age_min=40.0,
                age_max=70.0,
                n_participants=30,
                n_age_bins=3,
            ),
        ),
    )
    participants = pd.DataFrame(
        {
            "RIAGENDR": [1, 1],
            "RIDAGEYR": [50, 50],
            "participant_probability": [0.0, 1.0],
        }
    )

    mapped = apply_motionage_mapping(participants, mapping)

    assert mapped["MotionAge"].tolist() == [40.0, 70.0]
    assert mapped["MotionAgeAccel"].tolist() == [-10.0, 20.0]


def test_unknown_sex_group_maps_to_missing_motionage() -> None:
    mapping, _ = fit_motionage_mapping(
        _participants(),
        fit_partitions=["train"],
        representative_probability="mean",
        clip_eps=1.0e-4,
        weighted_fit=False,
        clamp_output_to_fit_age_range=False,
        require_positive_beta=True,
    )
    participants = pd.DataFrame(
        {
            "RIAGENDR": [9],
            "RIDAGEYR": [50],
            "participant_probability": [0.2],
        }
    )

    mapped = apply_motionage_mapping(participants, mapping)

    assert np.isnan(mapped.loc[0, "MotionAge"])
    assert np.isnan(mapped.loc[0, "MotionAgeAccel"])


def test_fit_motionage_mapping_rejects_non_positive_beta_when_required() -> None:
    participants = pd.DataFrame(
        {
            "split": ["train", "train", "train"],
            "RIAGENDR": [1, 1, 1],
            "RIDAGEYR": [40, 50, 60],
            "participant_probability": [0.6, 0.4, 0.2],
        }
    )

    with pytest.raises(ValueError, match="positive beta"):
        fit_motionage_mapping(
            participants,
            fit_partitions=["train"],
            representative_probability="mean",
            clip_eps=1.0e-4,
            weighted_fit=False,
            clamp_output_to_fit_age_range=False,
            require_positive_beta=True,
        )


def test_mapping_to_jsonable_preserves_public_parameters() -> None:
    mapping, _ = fit_motionage_mapping(
        _participants(),
        fit_partitions=["train"],
        representative_probability="mean",
        clip_eps=1.0e-4,
        weighted_fit=True,
        clamp_output_to_fit_age_range=False,
        require_positive_beta=True,
    )

    payload = mapping_to_jsonable(mapping)

    assert payload["method"] == "sex_specific_logistic_inverse"
    assert payload["fit_partitions"] == ["train"]
    assert payload["clip_eps"] == 1.0e-4
    assert payload["weighted_fit"] is True
    assert {group["sex_value"] for group in payload["groups"]} == {1, 2}
