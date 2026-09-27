"""Evaluate participant probabilities with optional AUROC uncertainty."""

from motionage.evaluation.workflow import run_evaluation
from motionage.workflow_io import config_main

if __name__ == "__main__":
    config_main(run_evaluation, __doc__)
