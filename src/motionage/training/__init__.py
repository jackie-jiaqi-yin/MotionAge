"""Training runtime helpers for MotionAge experiments."""

from motionage.training.checkpoint import build_checkpoint_payload
from motionage.training.device import mps_diagnostics, resolve_device
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
from motionage.training.loss import build_loss_function
from motionage.training.limits import optional_positive_int
from motionage.training.log_rows import (
    build_training_log_row,
    nan_validation_metrics,
    training_log_fieldnames,
    write_training_log_csv,
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
from motionage.training.scheduler import (
    build_optimizer_and_scheduler,
    reset_plateau_scheduler_for_stage_transition,
    restore_frozen_group_lrs,
)
from motionage.training.selection import initial_best_metric, metric_improved
from motionage.training.summary import (
    build_fixed_epoch_training_summary,
    build_training_summary,
)
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
    "build_checkpoint_payload",
    "build_fixed_epoch_training_summary",
    "build_loss_function",
    "build_model_inputs",
    "build_optimizer_and_scheduler",
    "build_optimizer_parameter_groups",
    "build_training_log_row",
    "build_training_summary",
    "build_trainval_refit_splits",
    "configure_torch_cpu_threads",
    "count_trainable_parameters",
    "is_binary_classification",
    "initial_best_metric",
    "metric_improved",
    "mps_diagnostics",
    "nan_validation_metrics",
    "optional_positive_int",
    "parse_freeze_schedule",
    "predict",
    "resolve_device",
    "resolve_final_refit_config",
    "resolve_freeze_stage",
    "resolve_task_type",
    "resolve_winner_trial",
    "reset_plateau_scheduler_for_stage_transition",
    "restore_frozen_group_lrs",
    "scheduler_mode",
    "selection_metric_direction",
    "selection_metric_name",
    "threshold_metric_name",
    "training_log_fieldnames",
    "write_training_log_csv",
]
