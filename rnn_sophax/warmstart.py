import logging
import pathlib
from typing import Optional, Tuple

import cloudpickle
import haiku as hk
from rich.panel import Panel

from rnn_sophax.fit_multi import TrainState

log = logging.getLogger(__file__)


class WarmstartType:
    Point = 0
    StartFrom = 1
    Resume = 2


WarmstartInfo = Tuple[pathlib.Path, WarmstartType]


def freeze_pointmass_params(train_state: TrainState, pointmass_train_state: Optional[TrainState]) -> TrainState:
    # We want to
    #   1: Remove all point_V_net params from train_state
    #   2: Extract point_V_net params from pointmass_params, put it in train_state.
    def is_pt_weight(module_name: str, name: str, value) -> bool:
        return "point_V_net" in module_name

    def is_norm_param(module_name: str, name: str, value) -> bool:
        return "normalizer" in module_name

    def is_frozen_param(module_name: str, name: str, value) -> bool:
        return is_pt_weight(module_name, name, value) or is_norm_param(module_name, name, value)

    freeze_params, train_params = hk.data_structures.partition(is_frozen_param, train_state.params)

    if pointmass_train_state is not None:
        pt_params, _ = hk.data_structures.partition(is_frozen_param, pointmass_train_state.params)
        freeze_params = hk.data_structures.merge(pt_params, pointmass_train_state.frozen_params)

    all_param_names = str(list(train_state.params.keys()))
    train_param_names = str(list(train_params.keys()))
    freeze_param_names = str(list(freeze_params.keys()))

    print(Panel(all_param_names, title="All Params", expand=False))
    print(Panel(train_param_names, title="Train Params", expand=False))
    print(Panel(freeze_param_names, title="Freeze Params", expand=False))
    train_state = train_state._replace(params=train_params, frozen_params=freeze_params)

    return train_state


def handle_warmstart(train_state: TrainState, warmstart: Optional[WarmstartInfo]) -> Tuple[TrainState, int]:
    if warmstart is None:
        log.info("No warmstart, training from scratch!")
        return train_state, 0

    path, warmstart_type = warmstart

    assert path.exists()

    with open(path, "rb") as f:
        ckpt = cloudpickle.load(f)

    if isinstance(ckpt, TrainState):
        ckpt_train_state = ckpt
        step = 0
    elif isinstance(ckpt, dict):
        assert "train_state" in ckpt
        ckpt_train_state = ckpt["train_state"]
        step = ckpt["step"]
    else:
        raise RuntimeError("???")

    if warmstart_type == WarmstartType.Point:
        log.info("Warmstarting using pointmass potential {}, step=0!".format(path))
        train_state = freeze_pointmass_params(train_state, ckpt_train_state)
        return train_state, step
    elif warmstart_type == WarmstartType.StartFrom:
        log.info("Starting training from {}, step=0!".format(path))
        return ckpt_train_state, 0
    elif warmstart_type == WarmstartType.Resume:
        log.info("Resuming training from {}, step={}!".format(path, step))
        assert step > 0, "Trying to resume from step = 0!"
        return ckpt_train_state, step
    else:
        raise RuntimeError("???")
