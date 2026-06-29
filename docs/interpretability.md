# Interpretability

MotionAge interpretability reports should distinguish phenotype-level summaries
from causal feature-importance claims.

## Activity Profiles

The `motionage.analysis.activity_profiles` module summarizes wear-retained
activity profiles by chronological-age band and MotionAge acceleration stratum.
It computes participant-level quantities first, then averages those quantities
within each group:

- mean wear-retained activity,
- weekly peak-to-trough amplitude,
- active hour count,
- daytime and nighttime activity means,
- daytime to nighttime ratio.

`compare_activity_profile_strata` reports high-minus-low MotionAge acceleration
contrasts within each chronological-age band. These tables are suitable for
publication-facing activity-pattern interpretation and for figure companions.
They should not be described as causal feature attribution.
