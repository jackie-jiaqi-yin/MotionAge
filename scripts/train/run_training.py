"""Train one binary activity model from a configuration."""

from motionage.training.workflow import run_training
from motionage.workflow_io import config_main

if __name__ == "__main__":
    config_main(run_training, __doc__)
