"""Shared column names for public MotionAge data utilities."""

from __future__ import annotations


class ActivityColumns:
    """Canonical columns used by NHANES activity data preparation."""

    ID = "SEQN"
    DAY = "PAXDAY"
    HOUR = "PAXHOUR"
    MINUTE = "PAXMINUT"
    MINUTE_INDEX = "PAXN"
    INTENSITY = "PAXINTEN"
    ATTENTION_FLAG = "attention_flag"
