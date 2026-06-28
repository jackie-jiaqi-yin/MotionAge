"""Training runtime helpers for MotionAge experiments."""

from motionage.training.evaluate import (
    aggregate_to_participant,
    align_meta_to_predictions,
    predict,
)
from motionage.training.freeze import (
    FreezeStage,
    apply_freeze_stage,
    count_trainable_parameters,
    parse_freeze_schedule,
    resolve_freeze_stage,
)
from motionage.training.model_inputs import build_model_inputs
from motionage.training.optimizer import (
    OptimizerGroupSummary,
    build_optimizer_parameter_groups,
)
from motionage.training.refit import (
    build_trainval_refit_splits,
    resolve_final_refit_config,
    resolve_winner_trial,
)
from motionage.training.runtime import configure_torch_cpu_threads
from motionage.training.task import (
    BINARY_CLASSIFICATION,
    REGRESSION,
    is_binary_classification,
    resolve_task_type,
    scheduler_mode,
    selection_metric_direction,
    selection_metric_name,
    threshold_metric_name,
)

__all__ = [
    "BINARY_CLASSIFICATION",
    "FreezeStage",
    "OptimizerGroupSummary",
    "REGRESSION",
    "aggregate_to_participant",
    "align_meta_to_predictions",
    "apply_freeze_stage",
    "build_model_inputs",
    "build_optimizer_parameter_groups",
    "build_trainval_refit_splits",
    "configure_torch_cpu_threads",
    "count_trainable_parameters",
    "is_binary_classification",
    "parse_freeze_schedule",
    "predict",
    "resolve_final_refit_config",
    "resolve_freeze_stage",
    "resolve_task_type",
    "resolve_winner_trial",
    "scheduler_mode",
    "selection_metric_direction",
    "selection_metric_name",
    "threshold_metric_name",
]
