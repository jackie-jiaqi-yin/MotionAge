"""Compute MotionAge and general evaluation from local participant predictions."""

from motionage.analysis.motionage.pipeline import run_motionage
from motionage.workflow_io import config_main

if __name__ == "__main__":
    config_main(run_motionage, __doc__)
