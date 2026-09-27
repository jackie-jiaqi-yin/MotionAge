"""Apply a saved model and training-fitted preprocessing to local inputs."""

from motionage.training.inference import run_prediction
from motionage.workflow_io import config_main

if __name__ == "__main__":
    config_main(run_prediction, __doc__)
