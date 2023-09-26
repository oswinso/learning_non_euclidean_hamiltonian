import logging
import pathlib
import pickle
import time
from typing import NamedTuple, Optional, Tuple

import haiku as hk
import jax
import optax
from jax import numpy as jnp

from integrators.symplectic import IntegrateFn, IntResult, get_verlet
from loggers.base_logger import BaseLogger
from loggers.logger_types import LogKey as LK
from rnn_sophax.loss_weight_schedule import LossWeightCfg, get_loss_weight_schedule
from models.so2rnn import TransformedSO2RNN, get_so2rnn_model
from rnn_sophax.optim import OptimCfg, get_optimizer
from rnn_sophax.so2_dset import PendDSet
from utils.clipping import get_global_norm
from utils.haiku_utils import Activation, MLPCfg
from utils.types import ja
from rnn_sophax.weighted_loss import get_loss_weights

log = logging.getLogger(__file__)


class LossAux(NamedTuple):
    sum_p_loss: ja
    sum_R_loss: ja
    pred_trajs: IntResult


class TrainState(NamedTuple):
    params: optax.Params
    opt_state: optax.InjectHyperparamsState


class StepResult(NamedTuple):
    train_state: TrainState
    loss: ja
    loss_aux: LossAux


def get_model() -> TransformedSO2RNN:
    V_cfg = MLPCfg(hidden=[16, 16], act=Activation.Tanh)
    g_cfg = MLPCfg(hidden=[1], act=Activation.Tanh)

    return get_so2rnn_model(V_cfg, g_cfg)


def get_step_fns(integrate: IntegrateFn, optimizer: optax.GradientTransformation, dset: PendDSet, jit: bool = True):
    def loss_fn(params: optax.Params, trajs: IntResult, loss_weights: Optional[ja] = None) -> Tuple[ja, LossAux]:
        # traj.Rs: (batch, 0)
        R0s, p_theta0s = trajs.Rs[:, 0], trajs.p_thetas[:, 0]
        pred_trajs = integrate(R0s, p_theta0s, params)

        assert pred_trajs.Rs.shape == trajs.Rs.shape
        # (batch, T)
        assert pred_trajs.p_thetas.shape == trajs.p_thetas.shape

        # (batch, T + 1)
        p_diff_sq = (pred_trajs.p_thetas - trajs.p_thetas) ** 2
        # (batch, T + 1, 2, 2)
        R_diff_sq = (pred_trajs.Rs - trajs.Rs) ** 2
        # (batch, T + 1)
        R_diff_sq = jnp.sum(R_diff_sq, axis=(2, 3))

        # Mean over batch.
        p_diff_sq = jnp.mean(p_diff_sq, axis=0)
        R_diff_sq = jnp.mean(R_diff_sq, axis=0)

        if loss_weights is not None:
            # Make sure the length matches.
            assert loss_weights.ndim == 1
            assert loss_weights.shape[0] == trajs.Rs.shape[1] and loss_weights.shape[0] == trajs.p_thetas.shape[1]

            weighted_p_diff_sq = p_diff_sq * loss_weights
            weighted_R_diff_sq = R_diff_sq * loss_weights
        else:
            weighted_p_diff_sq = p_diff_sq
            weighted_R_diff_sq = R_diff_sq

        # Sum over time.
        p_loss = jnp.sum(weighted_p_diff_sq)
        R_loss = jnp.sum(weighted_R_diff_sq)

        return p_loss + R_loss, LossAux(p_diff_sq, R_diff_sq, pred_trajs)

    def train_step(state: TrainState, trajs: IntResult, loss_weights: ja) -> StepResult:
        # Compute loss and gradients.
        (loss, loss_aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, trajs, loss_weights)

        # Gradient step.
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        return StepResult(TrainState(params=params, opt_state=opt_state), loss, loss_aux)

    def val_step(state: TrainState) -> StepResult:
        loss, loss_aux = loss_fn(state.params, dset.trajs)

        return StepResult(state, loss, loss_aux)

    if jit:
        log.info("Jitting train_step and val_step...")
        train_step = jax.jit(train_step)
        val_step = jax.jit(val_step)

    return train_step, val_step


def init_train_state(model: TransformedSO2RNN, optimizer: optax.GradientTransformation) -> TrainState:
    rng = hk.PRNGSequence(151)
    dummy_R = jnp.zeros((2, 2))
    dummy_Pi = jnp.zeros(1)
    params = model.init(next(rng), dummy_R, dummy_Pi)

    opt_state = optimizer.init(params)

    return TrainState(params, opt_state)


def log_train(
    logger: BaseLogger,
    idx: int,
    lr: float,
    final_step_weight: float,
    opt_state: optax.InjectHyperparamsState,
    loss: ja,
    loss_aux: LossAux,
    now: float,
    step_start_time: float,
    train_start_time: float,
) -> None:
    global_norm = get_global_norm(opt_state)
    entry = {
        LK.idx: idx,
        LK.lr: lr,
        # -----------------
        LK.final_step_weight: final_step_weight,
        # -----------------
        LK.loss: loss,
        LK.R_loss: loss_aux.sum_R_loss,
        LK.p_loss: loss_aux.sum_p_loss,
        # -----------------
        LK.grad_global_norm: global_norm,
        # -----------------
        LK.iter_time: now - step_start_time,
        LK.train_time: now - train_start_time,
    }
    logger.log_train(entry)


class FitResult(NamedTuple):
    model: TransformedSO2RNN
    train_state: TrainState

    def save(self, path: pathlib.Path) -> None:
        if path.suffix == "":
            path = path.with_suffix(".pkl")

        assert not path.exists(), "Trying to overwrite existing file {}!".format(path)
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def fit(dset: PendDSet, logger: BaseLogger) -> FitResult:
    model = get_model()

    integrate = get_verlet(model, dset.h, dset.n_steps, n_substeps=1)

    train_steps = 1024

    optim_cfg = OptimCfg(max_lr=3e-3, warmup_frac=0.1, final_lr_frac=0.1, grad_clip=1.0)
    optimizer, lr_scheduler = get_optimizer(train_steps, optim_cfg)

    train_step, val_step = get_step_fns(integrate, optimizer, dset, jit=True)
    train_state = init_train_state(model, optimizer)

    loss_weight_cfg = LossWeightCfg(base=2, max_power=12, steps=800)
    loss_weight_schedule = get_loss_weight_schedule(loss_weight_cfg)

    batch_size = 128
    dset_size = dset.trajs.p_thetas.shape[0]
    batches_per_epoch = dset_size // batch_size
    assert batches_per_epoch * batch_size == dset_size

    batch_idxs = jnp.zeros(1)

    rng = hk.PRNGSequence(182)

    step_start_time = train_start_time = time.time()
    for idx in range(train_steps):
        # Set the learning rate.
        lr = lr_scheduler(idx)
        train_state.opt_state.hyperparams["lr"] = lr

        # Sample data from dataset.
        batch_idx = idx % batches_per_epoch
        if batch_idx == 0:
            idxs = jax.random.permutation(next(rng), dset_size)
            batch_idxs = idxs.reshape(batches_per_epoch, batch_size)

        sample_idxs = batch_idxs[batch_idx]

        Rs, ps = dset.trajs.Rs[sample_idxs], dset.trajs.p_thetas[sample_idxs]

        # Get the weights for the loss.
        loss_weight_decay = loss_weight_schedule(idx)
        use_even_weights = idx > loss_weight_cfg.steps
        loss_weights = get_loss_weights(loss_weight_decay, dset.n_steps, use_even_weights)

        # Train Step.
        train_state, loss, loss_aux = train_step(train_state, IntResult(Rs, ps), loss_weights)
        now = time.time()

        # Log Train.
        log_train(
            logger,
            idx,
            lr,
            loss_weights[-1],
            train_state.opt_state,
            loss,
            loss_aux,
            now,
            step_start_time,
            train_start_time,
        )

    return FitResult(model, train_state)
