import enum
from typing import Any, Dict

import optax
from attr import s

# from solver.solver_impl import LossAux
from utils.types import ja

TrainLogData = Dict[str, Any]
ValLogData = Dict[str, Any]


class LogKey(str, enum.Enum):
    idx = "idx"
    lr = "lr"
    integrate_T = "integrate_T"
    # ----------------------------------
    train_state = "train_state"
    # ----------------------------------
    loss = "loss"
    q_loss = "q_loss"
    p_loss = "p_loss"
    R_loss = "R_loss"
    Pi_loss = "Pi_loss"
    # ----------------------------------
    Vdiff_mean = "Vdiff_mean"
    Vdiff_max = "Vdiff_max"
    # ----------------------------------
    q_gradnorm = "q_gradnorm"
    p_gradnorm = "p_gradnorm"
    R_gradnorm = "R_gradnorm"
    Pi_gradnorm = "Pi_gradnorm"
    # ----------------------------------
    final_step_weight = "final_step_weight"
    # ----------------------------------
    grad_global_norm_before = "grad_global_norm_before"
    grad_global_norm_after = "grad_global_norm_after"
    # ----------------------------------
    weight_norm = "weight_norm"
    # ----------------------------------
    iter_time = "iter_time"
    train_time = "train_time"
