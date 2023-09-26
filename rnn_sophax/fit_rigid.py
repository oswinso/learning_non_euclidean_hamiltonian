import pathlib
from typing import Optional, Tuple
import logging

import cloudpickle
import haiku as hk
import jax.numpy as jnp
import optax
from rich import print
from rich.panel import Panel

from grit_dset.planets_dset import PlanetsDSet
from integrators.integrator_enum import IntegratorEnum
from loggers.base_logger import BaseLogger
from models.multiso3rnn import TransformedSO3RNN
from rnn_sophax import weighted_loss as wl
from rnn_sophax.fit_multi import (
    FitResult,
    SolveCfg,
    TrainState,
    compute_and_set_normalization,
    fit,
    init_train_state,
)
from rnn_sophax.optim import OptimCfg, get_optimizer
from rnn_sophax.train_phase import (
    Phase,
    PhaseScheduler,
    trunc_lin_phase,
    truncated_phase,
    truncated_phase_onecycle,
    truncated_transition_phase,
)
from rnn_sophax.warmstart import WarmstartInfo, handle_warmstart

log = logging.getLogger(__file__)

def fit_rigid(
    model: TransformedSO3RNN,
    dset: PlanetsDSet,
    val_dset: PlanetsDSet,
    logger: BaseLogger,
    warmstart: Optional[WarmstartInfo],
    integrator: IntegratorEnum,
) -> FitResult:
    log.info("Using integrator {}!".format(integrator))

    phase_scheduler = get_rigid_scheduler_swish(dset.T, integrator)
    # phase_scheduler = get_rigid_scheduler_tanh(dset.T)
    # phase_scheduler = get_constant_scheduler(dset.T)

    if integrator == IntegratorEnum.VERLET:
        loss_coeff = 1.0
        solve_cfg = SolveCfg(loss_coeff=loss_coeff, q_coeff=1.0, p_coeff=1.0, R_coeff=1e5, Pi_coeff=6e5, val_every=50)
    else:
        loss_coeff = 1.0
        solve_cfg = SolveCfg(loss_coeff=loss_coeff, q_coeff=1.0, p_coeff=1.0, R_coeff=1e-4, Pi_coeff=6e5, val_every=50)

    optim_cfg = OptimCfg(grad_clip=1e-4)
    # solve_cfg = SolveCfg(loss_coeff=1.0, q_coeff=1.0, p_coeff=1.0, R_coeff=0.0, Pi_coeff=250.0, val_every=200)

    optimizer = get_optimizer(optim_cfg)
    model, train_state = init_train_state(dset.n_bodies, model, optimizer)

    # Freeze the passed in params.
    train_state, idx = handle_warmstart(train_state, warmstart)

    frozen_params = train_state.frozen_params
    n_RJR_params = dset.n_bodies * 3 * 3 * 2

    # log.debug("DEBUG! Overriding normalization from before.")
    train_state = compute_and_set_normalization(train_state, dset)

    # Reinit the optimizer since we changed the trainiable params.
    train_state = train_state._replace(opt_state=optimizer.init(train_state.params))

    return fit(model, train_state, optimizer, dset, val_dset, phase_scheduler, solve_cfg, logger, idx)


def get_rigid_scheduler_swish(T: int, integrator: IntegratorEnum) -> PhaseScheduler:
    if integrator == IntegratorEnum.VERLET:
        c = 2.0
    elif integrator == IntegratorEnum.EULER:
        c = 1.0
    elif integrator == IntegratorEnum.RK4:
        c = 1.0
    else:
        raise RuntimeError("...")

    init_max_lr = 8e-4 * c
    final_max_lr = 1e-6 * c

    max_power = 64
    n_warmup_steps = 1000
    transition_warmup_steps = 1000
    final_lr_frac = 0.35

    def transition_params(max_lr: float):
        return dict(
            max_power=max_power,
            T=T,
            max_lr=max_lr,
            n_warmup_steps=transition_warmup_steps,
            final_lr_frac=final_lr_frac,
        )

    phases1 = [
        # trunc_lin_phase(10000, 2, T, 1.0, init_max_lr, n_warmup_steps),
        # trunc_lin_phase(10000, 2, T, 1.0, init_max_lr / 5, n_warmup_steps),
        # trunc_lin_phase(10000, 2, T, 1.0, init_max_lr / 5 ** 2, n_warmup_steps),
        truncated_phase(10000, 2, T, 1.0, init_max_lr, n_warmup_steps=n_warmup_steps, final_lr_frac=0.3),
        truncated_transition_phase(10000, 2, 4, preclip_lr=5e-1, **transition_params(0.2 * init_max_lr)),
        truncated_transition_phase(10000, 4, 6, preclip_lr=5e-1, **transition_params(0.2 ** 2 * init_max_lr)),
        truncated_transition_phase(10000, 6, 8, preclip_lr=5e-2, **transition_params(0.2 ** 2 * init_max_lr)),
        truncated_transition_phase(10000, 8, 16, preclip_lr=5e-3, **transition_params(0.2 ** 3 * init_max_lr)),
        # truncated_transition_phase(10000, 16, 24, 5e-5, **transition_params(0.2 ** 4 * init_max_lr)),
        # truncated_transition_phase(10000, 24, 32, 5e-6, **transition_params(0.2 ** 4 * init_max_lr)),
        # truncated_transition_phase(10000, 32, T, 5e-7, **transition_params(0.2 ** 4 * init_max_lr)),
        # truncated_phase(10000, T, T, 1e-8, 0.2 ** 5 * init_max_lr, n_warmup_steps=1, final_lr_frac=0.0),
    ]
    phases2 = [
        truncated_phase(5000, 2, T, 1.0, init_max_lr, n_warmup_steps=n_warmup_steps),
        truncated_phase(5000, 2, T, 1.0, 0.9 * init_max_lr, n_warmup_steps=n_warmup_steps),
        truncated_phase(5000, 4, T, 5e-1, 5e-4, n_warmup_steps=1500, final_lr_frac=0.0),
        truncated_phase(5000, 4, T, 5e-1, 5e-4, n_warmup_steps=1500, final_lr_frac=0.0),
        truncated_phase(5000, 4, T, 5e-1, 5e-4, n_warmup_steps=1500, final_lr_frac=0.0),
        truncated_phase(5000, 4, T, 5e-1, 5e-4, n_warmup_steps=1500, final_lr_frac=0.0),
    ]
    phases3 = [trunc_lin_phase(10000, 1, T, 1.0, init_max_lr, n_warmup_steps)]

    phases4 = [
        truncated_phase_onecycle(1000, 2, T, preclip_lr=1.0, max_lr=4e-3),
        truncated_phase_onecycle(2000, 2, T, preclip_lr=1.0, max_lr=4e-3),
        truncated_phase_onecycle(4000, 2, T, preclip_lr=1.0, max_lr=4e-3),
        truncated_phase_onecycle(8000, 2, T, preclip_lr=1.0, max_lr=4e-3),
        truncated_phase_onecycle(20000, 2, T, preclip_lr=1.0, max_lr=4e-3),
        # truncated_phase_onecycle(50000, 2, T, preclip_lr=1.0, max_lr=2e-3),
        truncated_phase_onecycle(50000, 4, T, preclip_lr=2e-1, max_lr=2e-3),
        truncated_phase_onecycle(50000, 8, T, preclip_lr=1e-1, max_lr=5e-4),
        truncated_phase_onecycle(50000, 12, T, preclip_lr=1e-2, max_lr=5e-4),
        truncated_phase_onecycle(50000, 16, T, preclip_lr=1e-3, max_lr=5e-4),
        truncated_phase_onecycle(50000, 26, T, preclip_lr=1e-4, max_lr=5e-4),
        truncated_phase_onecycle(50000, 32, T, preclip_lr=1e-5, max_lr=5e-4),
    ]

    phases5 = [
        truncated_phase_onecycle(2000, 8, T, preclip_lr=1.0, max_lr=5e-4),
        truncated_phase_onecycle(4000, 8, T, preclip_lr=1.0, max_lr=5e-4),
        truncated_phase_onecycle(8000, 8, T, preclip_lr=1.0, max_lr=4e-4),
        truncated_phase_onecycle(16000, 8, T, preclip_lr=1.0, max_lr=3e-4),
        truncated_phase_onecycle(32000, 8, T, preclip_lr=0.9, max_lr=2e-4),
        truncated_phase_onecycle(50000, 12, T, preclip_lr=0.8, max_lr=1e-4),
        truncated_phase_onecycle(50000, 16, T, preclip_lr=0.6, max_lr=8e-5),
        truncated_phase_onecycle(50000, 32, T, preclip_lr=0.4, max_lr=5e-5),
        truncated_phase_onecycle(50000, T, T, preclip_lr=0.1, max_lr=1e-5),
    ]

    phases6 = [
        truncated_phase_onecycle(64000, 8, T, preclip_lr=1.0, max_lr=5e-4),
        truncated_phase_onecycle(50000, 12, T, preclip_lr=0.8, max_lr=1e-4),
        truncated_phase_onecycle(50000, 16, T, preclip_lr=0.8, max_lr=8e-5),
    ]

    phases = phases6

    return PhaseScheduler(phases)


def get_rigid_scheduler_tanh(T: int) -> PhaseScheduler:
    c = 1.0
    init_max_lr = 8e-4 * c
    final_max_lr = 8e-6 * c

    max_power = 64
    warmup_frac, transition_warmup_frac = 0.1, 0.1
    final_lr_frac = 0.1

    def transition_params(max_lr: float):
        return dict(
            max_power=max_power,
            T=T,
            max_lr=max_lr,
            warmup_frac=transition_warmup_frac,
            final_lr_frac=final_lr_frac,
        )

    phases = [
        truncated_phase(2000, 2, T, init_max_lr, warmup_frac, final_lr_frac=0.6),
        # truncated_transition_phase(2000, 2, 4, **transition_params(1.5e-4 * c)),
        # truncated_transition_phase(2000, 4, 8, **transition_params(1e-4 * c)),
        # truncated_transition_phase(1000, 8, 16, **transition_params(5e-5 * c)),
        # truncated_transition_phase(1000, 16, 32, **transition_params(5e-5 * c)),
        # truncated_transition_phase(1000, 32, T, **transition_params(5e-5 * c)),
        # truncated_phase(3000, T, T, final_max_lr, warmup_frac, final_lr_frac=0.0),
    ]
    return PhaseScheduler(phases)
