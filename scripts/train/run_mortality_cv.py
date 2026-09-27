"""Run configured mortality model cross-validation or one model configuration."""

from motionage.training.cross_validation import run_cross_validation
from motionage.workflow_io import config_main

if __name__ == "__main__":
    config_main(run_cross_validation, __doc__)
