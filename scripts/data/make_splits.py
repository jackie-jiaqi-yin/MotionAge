"""Generate stratified participant train/validation/test or CV splits."""

from motionage.data.partitioning import make_splits
from motionage.workflow_io import config_main

if __name__ == "__main__":
    config_main(make_splits, __doc__)
