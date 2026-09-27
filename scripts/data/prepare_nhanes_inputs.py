"""Prepare local NHANES inputs from an explicit configuration."""

from motionage.preprocessing.pipeline import prepare_inputs
from motionage.workflow_io import config_main

if __name__ == "__main__":
    config_main(prepare_inputs, __doc__)
