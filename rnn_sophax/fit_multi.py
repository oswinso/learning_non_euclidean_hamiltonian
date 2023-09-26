import logging
import math
import time
from typing import Any, Dict, List, NamedTuple, Tuple, TypeVar

import haiku as hk
import ipdb
import jax
import numpy as np
import optax
from attr import s
from jax import numpy as jnp

from grit_dset.planets_dset import PlanetsDSet
from integrators.multi_euler import get_multi_euler
from integrators.multi_symplectic import IntegrateFn, IntResult, get_multi_verlet
from loggers.base_logger import BaseLogger
from loggers.logger_types import LogKey as LK
from models.input_normalizer import set_normalization
from models.multiso3rnn import MultiSO3RNN, TransformedSO3RNN, make_nn_input
from rnn_sophax import weighted_loss as wl
from rnn_sophax.optim import OptimCfg, get_optimizer, not_bias
from rnn_sophax.train_phase import (
    PhaseScheduler,
    truncated_phase,
    truncated_transition_phase,
)
from utils.clipping import get_global_norm
from utils.jax_utils import global_norm, value_and_jacrev
from utils.lie import cos_geodesic
from utils.types import ja

_T = TypeVar("_T")

log = logging.getLogger(__file__)


class LossAux(NamedTuple):
    sum_q_loss: ja
    sum_p_loss: ja
    sum_R_loss: ja
    sum_Pi_loss: ja
    pred_trajs: IntResult
    # Gradient norms of [q, p, R, Pi] AFTER WEIGHTING!
    weighted_gradnorms: List[ja]


class TrainState(NamedTuple):
    params: optax.Params
    frozen_params: optax.Params
    opt_state: optax.InjectHyperparamsState


class StepResult(NamedTuple):
    train_state: TrainState
    loss: ja
    loss_aux: LossAux
    grads: Any


class ValResult(NamedTuple):
    loss: ja
    loss_aux: LossAux
    potential_metrics: Dict[str, ja]


@s(auto_attribs=True, slots=True, auto_exc=True, auto_detect=True, order=False)
class SolveCfg:
    loss_coeff: float
    q_coeff: float
    p_coeff: float
    R_coeff: float
    Pi_coeff: float

    val_every: int


def l1_loss(diff: ja) -> ja:
    return jnp.abs(diff)


def l2_loss(diff: ja) -> ja:
    return 0.5 * (diff ** 2)


def huber_loss(diff: ja, delta: float) -> ja:
    return optax.huber_loss(diff, delta=delta)


def get_q(x: IntResult):
    return x.qs


def get_p(x: IntResult):
    return x.ps


def get_Pi(x: IntResult):
    return x.Pis


def compute_q_loss(_pred_trajs: IntResult, _true_trajs: IntResult):
    return compute_vec_diff_sq(get_q)(_pred_trajs, _true_trajs)


def compute_p_loss(_pred_trajs: IntResult, _true_trajs: IntResult):
    return compute_vec_diff_sq(get_p)(_pred_trajs, _true_trajs)


def compute_Pi_loss(_pred_trajs: IntResult, _true_trajs: IntResult):
    return compute_vec_diff_sq(get_Pi)(_pred_trajs, _true_trajs)


def compute_R_loss(_pred_trajs: IntResult, _true_trajs: IntResult):
    return compute_R_diff_sq()(_pred_trajs, _true_trajs)


def _compute_weighted_loss(
    _pred_trajs: IntResult, _true_trajs: IntResult, loss_weights: ja, time_weights: ja
) -> Tuple[ja, LossAux]:
    aux = compute_loss_components(_pred_trajs, _true_trajs)

    # (4, T + 1)
    losses = jnp.stack([aux.sum_q_loss, aux.sum_p_loss, aux.sum_R_loss, aux.sum_Pi_loss], axis=0)

    # Weigh by time.
    assert losses.shape[1] == time_weights.shape[0]

    # (4, )
    losses = jnp.sum(losses * time_weights, axis=1)

    loss = losses * loss_weights

    # Return a vector and DONT SUM so we have the gradients for each loss component.
    assert loss.shape == (4,)

    return loss, aux


def compute_loss_components(_pred_trajs: IntResult, _true_trajs: IntResult) -> LossAux:
    # All of these should have shape (T + 1, )
    q_loss = compute_q_loss(_pred_trajs, _true_trajs)
    p_loss = compute_p_loss(_pred_trajs, _true_trajs)
    R_loss = compute_R_loss(_pred_trajs, _true_trajs)
    Pi_loss = compute_Pi_loss(_pred_trajs, _true_trajs)

    loss_aux = LossAux(q_loss, p_loss, R_loss, Pi_loss, _pred_trajs, [])

    return loss_aux


def compute_vec_diff_sq(extract):
    def _loss(_pred_trajs: IntResult, _true_trajs: IntResult) -> ja:
        T_p_1 = _pred_trajs.qs.shape[1]
        pred_qty = extract(_pred_trajs)
        true_qty = extract(_true_trajs)[:, :T_p_1]

        diff_sq = l2_loss(pred_qty - true_qty)
        # Sum over n_bodies and data dim -> (batch, T + 1).
        diff_sq = jnp.sum(diff_sq, axis=(2, 3))
        # Mean over batch -> (T+1, )
        diff_sq = jnp.mean(diff_sq, axis=0)

        # Time dim should be left. (T + 1, )
        assert diff_sq.ndim == 1

        return diff_sq

    return _loss


def compute_R_diff_sq():
    def _loss(_pred_trajs: IntResult, _true_trajs: IntResult) -> ja:
        T_p_1 = _pred_trajs.qs.shape[1]

        # Use l2, since geodesic norm behaves pretty poorly when the error is ~0 due to acos behaving badly around 1.
        # R_diff = _pred_trajs.Rs - _true_trajs.Rs[:, :T_p_1]

        # Do avoid acos, do l2 loss on | cos(geodesic) - 1|^2. We want cos(geodesic) = 1 => geodesic = 0.
        R_diff = cos_geodesic(_pred_trajs.Rs, _true_trajs.Rs[:, :T_p_1]) - 1

        diff_sq = l2_loss(R_diff)

        # Sum over n_bodies and data dim -> (batch, T + 1).
        if diff_sq.shape[-2:] == (3, 3):
            diff_sq = jnp.sum(diff_sq, axis=(2, 3, 4))
        else:
            diff_sq = jnp.sum(diff_sq, axis=2)

        # Mean over batch -> (T+1, )
        diff_sq = jnp.mean(diff_sq, axis=0)

        # Time dim should be left. (T + 1, )
        assert diff_sq.ndim == 1

        return diff_sq

    return _loss


def get_step_fns(
    model: TransformedSO3RNN,
    integrate: IntegrateFn,
    optimizer: optax.GradientTransformation,
    cfg: SolveCfg,
    jit: bool = True,
):
    # jit_integrate = jax.jit(integrate)
    jit_integrate = integrate
    jit_integrate = jax.profiler.annotate_function(jit_integrate, name="jit_integrate")

    @jax.profiler.annotate_function
    def compute_weighted_loss(params: optax.Params, frozen_params: optax.Params, trajs: IntResult, time_weights: ja):
        all_params = hk.data_structures.merge(params, frozen_params, check_duplicates=True)

        # traj.Rs: (batch, 0)
        q0s, p0s, R0s, Pi0s = trajs.qs[:, 0], trajs.ps[:, 0], trajs.Rs[:, 0], trajs.Pis[:, 0]
        pred_trajs = jit_integrate(q0s, p0s, R0s, Pi0s, all_params)

        loss_weights = cfg.loss_coeff * jnp.array([cfg.q_coeff, cfg.p_coeff, cfg.R_coeff, cfg.Pi_coeff])
        return _compute_weighted_loss(pred_trajs, trajs, loss_weights, time_weights)

    def compute_weighted_loss_scalar(
        params: optax.Params, frozen_params: optax.Params, trajs: IntResult, time_weights: ja
    ):
        loss, aux = compute_weighted_loss(params, frozen_params, trajs, time_weights)
        return loss.sum(), aux

    def get_loss_and_jac(state: TrainState, trajs: IntResult, time_weights: ja) -> Tuple[Tuple[ja, LossAux], hk.Params]:
        # jac: Pytree of (4, *).
        aux: LossAux
        (loss_val, aux), jac = value_and_jacrev(compute_weighted_loss, has_aux=True)(
            state.params, state.frozen_params, trajs, time_weights
        )

        # Sum over the 4 components.
        sum_grads = jax.tree_map(lambda x: jnp.sum(x, axis=0), jac)

        # Separate the grad norms in jac.
        grad_norms_sq = sum([jnp.sum(jnp.square(x).reshape((4, -1)), axis=1) for x in jax.tree_leaves(jac)])
        grad_norms = jnp.sqrt(grad_norms_sq)

        assert grad_norms.shape == (4,)
        aux = aux._replace(weighted_gradnorms=grad_norms)

        # Return a scalar loss - sum over all components.
        sum_loss = loss_val.sum()

        return (sum_loss, aux), sum_grads

    def get_value_and_grad(
        state: TrainState, trajs: IntResult, time_weights: ja
    ) -> Tuple[Tuple[ja, LossAux], optax.Params]:
        (loss, loss_aux), grads = jax.value_and_grad(compute_weighted_loss_scalar, has_aux=True)(
            state.params, state.frozen_params, trajs, time_weights
        )
        return (loss, loss_aux), grads

    @jax.profiler.annotate_function
    def train_step(state: TrainState, trajs: IntResult, time_weights: ja) -> StepResult:
        # Compute loss and gradients.
        loss_aux: LossAux
        (loss, loss_aux), grads = jax.value_and_grad(compute_weighted_loss_scalar, has_aux=True)(
            state.params, state.frozen_params, trajs, time_weights
        )

        # Gradient step.
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        return StepResult(
            TrainState(params=params, frozen_params=state.frozen_params, opt_state=opt_state), loss, loss_aux, grads
        )

    @jax.profiler.annotate_function
    def train_jac_step(state: TrainState, trajs: IntResult, time_weights: ja) -> StepResult:
        # Compute loss and gradients.
        (loss, loss_aux), grads = get_loss_and_jac(state, trajs, time_weights)

        # Gradient step.
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        return StepResult(
            TrainState(params=params, frozen_params=state.frozen_params, opt_state=opt_state), loss, loss_aux, grads
        )

    def compute_potential_metrics(params: hk.Params, trajs: IntResult) -> Dict[str, ja]:
        # qs: (batch, T, n_bodies, 3), Rs: (batch, T, n_bodies, 3, 3)
        qs, Rs = trajs.qs, trajs.Rs

        batch, T, n_bodies, _ = qs.shape

        flat_qs, flat_Rs = qs.reshape(batch * T, n_bodies, 3), Rs.reshape(batch * T, n_bodies, 3, 3)

        # (batch * T, )
        pred_V = jax.vmap(model.rigid_V, (None, 0, 0))(params, flat_qs, flat_Rs)
        true_V = jax.vmap(model.rigid_correction, (None, 0, 0))(params, flat_qs, flat_Rs)

        assert pred_V.shape == (batch * T,) and true_V.shape == (batch * T,)

        # Offset so the means are the same.
        offset = jnp.mean(true_V) - jnp.mean(pred_V)
        pred_V = pred_V + offset

        diff = jnp.abs(true_V - pred_V)

        mean_diff = diff.mean()
        max_diff = diff.max()

        return dict(mean=mean_diff, max=max_diff)

    @jax.profiler.annotate_function
    def val_step(state: TrainState, trajs: IntResult) -> ValResult:
        all_params = hk.data_structures.merge(state.params, state.frozen_params, check_duplicates=True)
        q0s, p0s, R0s, Pi0s = trajs.qs[:, 0], trajs.ps[:, 0], trajs.Rs[:, 0], trajs.Pis[:, 0]

        pred_trajs: IntResult
        pred_trajs = jit_integrate(q0s, p0s, R0s, Pi0s, all_params)

        Tp1 = pred_trajs.qs.shape[1]

        loss_weights = cfg.loss_coeff * jnp.array([cfg.q_coeff, cfg.p_coeff, cfg.R_coeff, cfg.Pi_coeff])
        time_weights = jnp.zeros(Tp1)
        time_weights = time_weights.at[1:].set(1.0 / (Tp1 - 1))

        val_loss, loss_aux = _compute_weighted_loss(pred_trajs, trajs, loss_weights, time_weights)
        val_loss = val_loss.sum()

        # Compare the learned potential function with the true potential function.
        potential_metrics = compute_potential_metrics(all_params, trajs)

        return ValResult(val_loss, loss_aux, potential_metrics)

    if jit:
        log.info("Jitting train_step and val_step...")
        train_step = jax.jit(train_step)
        train_jac_step = jax.jit(train_jac_step)
        val_step = jax.jit(val_step)

    return train_step, train_jac_step, val_step


def init_train_state(
    n_bodies: int,
    model: TransformedSO3RNN,
    optimizer: optax.GradientTransformation,
) -> Tuple[TransformedSO3RNN, TrainState]:
    rng = hk.PRNGSequence(151)
    dummy_q = jax.random.normal(next(rng), (n_bodies, 3))
    dummy_p = jax.random.normal(next(rng), (n_bodies, 3))
    dummy_R = jax.random.normal(next(rng), (n_bodies, 3, 3))
    dummy_Pi = jax.random.normal(next(rng), (n_bodies, 3))

    rng = hk.PRNGSequence(151)
    params, frozen_params, param_names = model.init(next(rng), dummy_q, dummy_p, dummy_R, dummy_Pi)
    model = model._replace(param_names=param_names)

    # Separate normalization params
    def is_normalizer(module_name: str, name: str, value) -> bool:
        return "normalizer" in module_name

    frozen_params, params = hk.data_structures.partition(is_normalizer, params)

    opt_state = optimizer.init(params)

    return model, TrainState(params, frozen_params, opt_state)


def log_train(
    logger: BaseLogger,
    idx: int,
    lr: float,
    integrate_T: int,
    final_step_weight: float,
    train_state: TrainState,
    loss: ja,
    loss_aux: LossAux,
    now: float,
    step_start_time: float,
    train_start_time: float,
) -> None:
    with jax.profiler.TraceAnnotation("log_train"):
        gn_before, gn_after = get_global_norm(train_state.opt_state)
        entry = {
            LK.idx: idx,
            LK.lr: lr,
            LK.integrate_T: integrate_T,
            # -----------------
            LK.train_state: train_state,
            # -----------------
            LK.final_step_weight: final_step_weight,
            # -----------------
            LK.loss: loss,
            LK.q_loss: loss_aux.sum_q_loss,
            LK.p_loss: loss_aux.sum_p_loss,
            LK.R_loss: loss_aux.sum_R_loss,
            LK.Pi_loss: loss_aux.sum_Pi_loss,
            # -----------------
            LK.grad_global_norm_before: gn_before,
            LK.grad_global_norm_after: gn_after,
            # -----------------
            LK.iter_time: now - step_start_time,
            LK.train_time: now - train_start_time,
        }
        if len(loss_aux.weighted_gradnorms) > 0:
            entry = {
                LK.q_gradnorm: loss_aux.weighted_gradnorms[0],
                LK.p_gradnorm: loss_aux.weighted_gradnorms[1],
                LK.R_gradnorm: loss_aux.weighted_gradnorms[2],
                LK.Pi_gradnorm: loss_aux.weighted_gradnorms[3],
                **entry,
            }
        # -----------------
        logger.log_train(entry)


def log_valid(logger: BaseLogger, idx: int, lr: float, train_state: TrainState, result: ValResult) -> None:
    # Compute the weight norm and log it.
    weights, _ = hk.data_structures.partition(not_bias, train_state.params)
    weight_norm = global_norm(weights)

    entry = {
        LK.idx: idx,
        # -------------------------
        LK.train_state: train_state,
        # --------------------------
        LK.loss: result.loss,
        LK.q_loss: result.loss_aux.sum_q_loss,
        LK.p_loss: result.loss_aux.sum_p_loss,
        LK.R_loss: result.loss_aux.sum_R_loss,
        LK.Pi_loss: result.loss_aux.sum_Pi_loss,
        # --------------------------
        LK.Vdiff_mean: result.potential_metrics["mean"],
        LK.Vdiff_max: result.potential_metrics["max"],
        # --------------------------
        LK.weight_norm: weight_norm,
    }
    logger.log_val(entry)


class FitResult(NamedTuple):
    model: TransformedSO3RNN
    train_state: TrainState


def get_pos_scheduler(T: int) -> PhaseScheduler:
    lr_mult = 1.0

    init_max_lr = 8e-4 * lr_mult
    final_max_lr = 8e-6 * lr_mult

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
            # grad_coeff=grad_coeff,
        )

    phases = [
        truncated_phase(3000, 2, T, init_max_lr, warmup_frac, final_lr_frac=0.3),
        truncated_transition_phase(1000, 2, 4, **transition_params(2e-4)),
        truncated_transition_phase(1000, 4, 8, **transition_params(2e-4)),
        truncated_transition_phase(1000, 8, 16, **transition_params(5e-5)),
        truncated_transition_phase(1000, 16, 32, **transition_params(5e-5)),
        truncated_transition_phase(1000, 32, T, **transition_params(5e-5)),
        truncated_phase(3000, 32, T, final_max_lr, warmup_frac, final_lr_frac=0.0),
    ]
    return PhaseScheduler(phases)


def fit_pointmass(model: TransformedSO3RNN, dset: PlanetsDSet, val_dset: PlanetsDSet, logger: BaseLogger) -> FitResult:
    phase_scheduler = get_pos_scheduler(dset.T)
    optim_cfg = OptimCfg(grad_clip=0.6)
    solve_cfg = SolveCfg(loss_coeff=1.0, q_coeff=1.0, p_coeff=1.0, R_coeff=0.0, Pi_coeff=0.0, val_every=200)

    optimizer = get_optimizer(optim_cfg)
    model, train_state = init_train_state(dset.n_bodies, model, optimizer)
    train_state = compute_and_set_normalization(train_state, dset)

    return fit(model, train_state, optimizer, dset, val_dset, phase_scheduler, solve_cfg, logger)


def get_constant_scheduler(T: int) -> PhaseScheduler:
    init_max_lr = 8e-4
    warmup_frac = 0.1
    phases = [
        truncated_phase(6000, 1, T, init_max_lr, warmup_frac, final_lr_frac=0.3),
    ]
    return PhaseScheduler(phases)


def compute_and_set_normalization(train_state: TrainState, dset: PlanetsDSet) -> TrainState:
    n_bodies = dset.n_bodies

    log.info("Computing normalization...")

    max_samples = 8192
    key = jax.random.PRNGKey(4421)

    dset_q = dset.q.reshape(-1, n_bodies, 3)
    dset_R = dset.R.reshape(-1, n_bodies, 3, 3)

    n_samples = min(max_samples, dset_q.shape[0])
    idxs = jax.random.choice(key, dset_q.shape[0], shape=(n_samples,), replace=False)
    dset_q, dset_R = dset_q[idxs], dset_R[idxs]

    fake_Js = jnp.expand_dims(MultiSO3RNN.get_true_Js(n_bodies), axis=0)
    fake_Js = jnp.broadcast_to(fake_Js, (dset_q.shape[0], n_bodies, 3, 3))

    V_input = jax.vmap(make_nn_input)(dset_q, dset_R, fake_Js)
    frozen_params = set_normalization(V_input, train_state.frozen_params)
    train_state = train_state._replace(frozen_params=frozen_params)
    del dset_q, dset_R, V_input

    return train_state


def fit(
    model: TransformedSO3RNN,
    train_state: TrainState,
    optimizer: optax.GradientTransformation,
    dset: PlanetsDSet,
    val_dset: PlanetsDSet,
    phase_scheduler: PhaseScheduler,
    solve_cfg: SolveCfg,
    logger: BaseLogger,
    idx: int = 0,
) -> FitResult:
    n_substeps = 1

    # get_integrator = get_multi_verlet
    get_integrator = get_multi_euler

    integrate_full = get_integrator(model, dset.dt, dset.T, n_substeps, vmap=True)
    integrator, step_fn = None, None

    step_fn_full = get_step_fns(model, integrate_full, optimizer, solve_cfg, jit=True)

    train_steps = phase_scheduler.get_train_steps()

    # step_fns = {t: get_step_fns(integrators[t], optimizer, solve_cfg, jit=True) for t in integrate_horizons}

    # Split data into chunks of consecutive trajectories.
    batch_size = 256
    dset_size = dset.batch
    batches_per_epoch = dset_size // batch_size

    # Truncate the dset.
    dset_size = batches_per_epoch * batch_size

    batch_idxs = None
    integrate_t = None

    rng = hk.PRNGSequence(182)

    n_val = 512
    # val_idxs = jax.random.permutation(next(rng), val_dset.batch)
    # val_idxs = val_idxs[:n_val]
    val_idxs = jnp.arange(n_val)

    # Put all the arrays in dset on the GPU.
    def transfer_to_gpu(x: _T) -> _T:
        gpu_device = jax.devices()[0]
        if isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray):
            return jax.device_put(x, gpu_device)

        return x

    dset = jax.tree_map(transfer_to_gpu, dset)

    val_qs, val_ps, val_Rs, val_Pis = dset.q[val_idxs], dset.p[val_idxs], dset.R[val_idxs], dset.Pi[val_idxs]
    val_trajs = IntResult(val_qs, val_ps, val_Rs, val_Pis)

    lr = 0
    step_start_time = train_start_time = time.time()
    for idx in range(idx, train_steps):
        with jax.profiler.StepTraceAnnotation("train", step_num=idx):
            # Set the learning rate.
            with jax.profiler.StepTraceAnnotation("set lr", step_num=idx):
                lr = phase_scheduler.get_lr(idx)
                train_state.opt_state.hyperparams["lr"] = lr

                # Set the pre-clip lr.
                preclip_lr = phase_scheduler.get_preclip_lr(idx)
                train_state.opt_state.hyperparams["preclip_lr"] = preclip_lr

            # Sample data from dataset.
            with jax.profiler.StepTraceAnnotation("Sample data", step_num=idx):
                batch_idx = idx % batches_per_epoch
                # batch_idx = 0
                if batch_idx == 0 or batch_idxs is None:
                    idxs = jax.random.permutation(next(rng), dset_size)
                    batch_idxs = idxs.reshape(batches_per_epoch, batch_size)

                sample_idxs = batch_idxs[batch_idx]

                qs, ps, Rs, Pis = dset.q[sample_idxs], dset.p[sample_idxs], dset.R[sample_idxs], dset.Pi[sample_idxs]

            # Get the weights for the loss.
            with jax.profiler.StepTraceAnnotation("Get loss weights", step_num=idx):
                loss_weights, first_zero = wl.get_loss_weights(idx, phase_scheduler.get_wl_sched(idx))

                if first_zero is not None:
                    new_integrate_t = first_zero - 1
                else:
                    new_integrate_t = dset.T

                if integrate_t is None or integrate_t != new_integrate_t:
                    log.info("Switching to integrate {:3} steps!".format(new_integrate_t))
                    integrate_t = new_integrate_t

                    if integrate_t < dset.T:
                        # Delete to try and clear memory.
                        integrator = get_integrator(model, dset.dt, integrate_t, n_substeps, vmap=True)
                        step_fn = get_step_fns(model, integrator, optimizer, solve_cfg, jit=True)
                    else:
                        step_fn = step_fn_full

            train_step, train_jac_step, _ = step_fn
            truncated_loss_weights = loss_weights[: integrate_t + 1]

            if idx % 100 == 0:
                log.info("Loss weights: {}".format(truncated_loss_weights))

            # Train Step.
            step_result: StepResult

            # Compute Jacobians when we validate, otherwise only compute gradients to try and speed up training.
            if False and idx % solve_cfg.val_every == 0:
                step_result = train_jac_step(train_state, IntResult(qs, ps, Rs, Pis), truncated_loss_weights)
            else:
                step_result = train_step(train_state, IntResult(qs, ps, Rs, Pis), truncated_loss_weights)

            now = time.time()

            train_state = step_result.train_state

            # Log Train. Don't do it every iteration since D2H takes a lot of time.
            if idx % 5 == 0:
                log_train(
                    logger,
                    idx,
                    lr,
                    integrate_t,
                    loss_weights[-1],
                    train_state,
                    step_result.loss,
                    step_result.loss_aux,
                    now,
                    step_start_time,
                    train_start_time,
                )
            step_start_time = now

            # Validation.
            if idx % solve_cfg.val_every == 0:
                with jax.profiler.TraceAnnotation("Validation"):
                    # TODO: Validate using EMA.
                    _, _, val_step = step_fn_full
                    val_result = val_step(train_state, val_trajs)
                    log_valid(
                        logger,
                        idx,
                        lr,
                        train_state,
                        val_result,
                    )
                    del val_result

            with jax.profiler.TraceAnnotation("check loss NaN"):
                if math.isnan(step_result.loss):
                    log.error("Loss is NaN! Exiting...")
                    break

    # Validate at the end.
    _, _, val_step = step_fn_full
    val_result = val_step(train_state, val_trajs)
    log_valid(
        logger,
        train_steps,
        lr,
        train_state,
        val_result,
    )

    return FitResult(model, train_state)
