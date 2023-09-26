import pathlib
from typing import Callable, NamedTuple, Protocol, Tuple, TypedDict

import chex
import cloudpickle
import haiku as hk
import ipdb
import jax
import jax.numpy as jnp
import optax
import torch
from jax import lax
from rich import print
from rich.progress import track

from utils.jax_utils import global_norm, scalarize
from utils.types import ja

PKL_PATH = pathlib.Path(__file__).parent / "train_dict.pkl"


class TrainData(TypedDict):
    train_data: torch.Tensor
    T: int
    batch_size: int
    n_epochs: int
    n_samples: int
    dt: int
    lr: float
    n_layers: int
    n_hidden: int
    test_data: torch.Tensor
    T_test: int
    T_init_seq: int


class MLP3HSep(hk.Module):
    def __init__(self, n_hidden: int):
        super().__init__()

        self.Ks = hk.nets.MLP([n_hidden, n_hidden, n_hidden, 1], activation=jax.nn.tanh)
        self.Ps = hk.nets.MLP([n_hidden, n_hidden, n_hidden, 1], activation=jax.nn.tanh)

    def ke(self, p: ja) -> ja:
        return self.Ks(p)

    def pe(self, q: ja) -> ja:
        return self.Ps(q)

    def energy(self, p: ja, q: ja) -> ja:
        return self.ke(p) + self.pe(q)


class TrainState(NamedTuple):
    params: optax.Params
    opt_state: optax.OptState


class StepResult(NamedTuple):
    loss: ja
    grad_norm: ja


def get_optim(lr: float) -> optax.GradientTransformation:
    return optax.adam(lr)


class EnergyFn(Protocol):
    def __call__(self, params: optax.Params, p: ja, q: ja) -> ja:
        ...


class SRNNModel(NamedTuple):
    init: Callable[[chex.PRNGKey], optax.Params]
    energy: EnergyFn


def get_model(n_hidden: int, dim: int) -> SRNNModel:
    def _fwd(p: ja, q: ja) -> ja:
        _model = MLP3HSep(n_hidden)
        return _model.energy(p, q)

    model = hk.without_apply_rng(hk.transform(_fwd))

    def _init(rng: chex.PRNGKey) -> optax.Params:
        p = jnp.ones(dim)
        q = jnp.ones(dim)
        return model.init(rng, p, q)

    return SRNNModel(_init, model.apply)


def init_state(model: SRNNModel, optimizer: optax.GradientTransformation) -> TrainState:
    rng = jax.random.PRNGKey(5452)
    params = model.init(rng)
    opt_state = optimizer.init(params)

    return TrainState(params, opt_state)


class OuterState(NamedTuple):
    q: ja
    p: ja


class IntState(NamedTuple):
    q: ja
    p: ja


def get_leapfrog(model: SRNNModel, dt: float, T: int, scan: bool):
    dHdp = jax.grad(scalarize(model.energy), argnums=1)
    dHdq = jax.grad(scalarize(model.energy), argnums=2)

    full_dt = dt
    half_dt = dt

    def _leapfrog_qfirst(params: optax.Params, p0: ja, q0: ja) -> ja:
        # p: (dim, ), q: (dim, )
        p, q = p0, q0
        (dim,) = p.shape

        ps = []
        qs = []

        dqdt = dHdp(params, p, q)

        for i in range(T):
            q_half = q + dqdt * half_dt
            # p_half = p + dpdt * half_dt

            ps.append(p)
            qs.append(q)

            dpdt = -dHdq(params, p, q)
            p_next = p + dpdt * full_dt
            # dqdt = dHdp(params, p, q)
            # q_next = q + dqdt * full_dt

            dqdt = dHdp(params, p, q)
            q_next = q_half + dqdt * half_dt
            # dpdt = -dHdq(params, p, q)
            # p_next = p_half + dpdt * half_dt

            p, q = p_next, q_next

        # (T, dim)
        p = jnp.stack(ps, axis=0)
        q = jnp.stack(qs, axis=0)

        # (T, 2 * dim)
        traj = jnp.concatenate([p, q], axis=1)
        assert traj.shape == (T, 2 * dim)

        return traj

    def _leapfrog_pfirst(params: optax.Params, p0: ja, q0: ja) -> ja:
        # p: (dim, ), q: (dim, )
        p, q = p0, q0
        (dim,) = p.shape

        ps = []
        qs = []

        dpdt = -dHdq(params, p, q)

        for i in range(T):
            p_half = p + dpdt * half_dt

            ps.append(p)
            qs.append(q)

            dqdt = dHdp(params, p, q)
            q_next = q + dqdt * full_dt

            dpdt = -dHdq(params, p, q)
            p_next = p_half + dpdt * half_dt

            p, q = p_next, q_next

        # (T, dim)
        p = jnp.stack(ps, axis=0)
        q = jnp.stack(qs, axis=0)

        # (T, 2 * dim)
        traj = jnp.concatenate([p, q], axis=1)
        assert traj.shape == (T, 2 * dim)

        return traj

    def get_outer_loop(params: hk.Params):
        def _outer_loop(carry, _) -> Tuple[OuterState, IntState]:
            dq = -dHdp(params, carry.p, carry.q)
            A_q = carry.q + half_dt * dq
            A_p = carry.p

            # Full step
            dp = dHdq(params, A_p, A_q)
            B_q = A_q
            B_p = A_p + full_dt * dp

            # Half step.
            dq = -dHdp(params, B_p, B_q)
            new_A_q = B_q + half_dt * dq
            new_A_p = B_p

            return OuterState(new_A_q, new_A_p), IntState(carry.q, carry.p)

        return _outer_loop

    def _leapfrog_scan(params: optax.Params, p0: ja, q0: ja) -> ja:
        (dim,) = p0.shape

        outer_loop = get_outer_loop(params)

        final_carry, outputs = lax.scan(outer_loop, OuterState(q0, p0), None, length=T)

        p, q = outputs.p, outputs.q
        traj = jnp.concatenate([p, q], axis=1)
        assert traj.shape == (T, 2 * dim)

        return traj

    if scan:
        return _leapfrog_scan
    else:
        return _leapfrog_qfirst
        # return _leapfrog_pfirst


def get_step(integrate, optimizer: optax.GradientTransformation, dim: int, max_t: int):
    # Don't vmap over parameters.
    vmap_integrate = jax.vmap(integrate, in_axes=(None, 0, 0), out_axes=1)

    def _loss(params: optax.Params, data: ja) -> ja:
        # batch: (T, batch, 2 * dim)
        T, batch, _ = data.shape

        z0_batch = data[0]
        p0 = z0_batch[:, :dim]
        q0 = z0_batch[:, dim:]

        assert p0.shape == (batch, dim)
        assert q0.shape == (batch, dim)

        traj_simulated = vmap_integrate(params, p0, q0)
        assert traj_simulated.shape[1:] == (batch, 2 * dim)

        pred = traj_simulated[:max_t, :, :]
        label = data[:max_t, :, :]

        assert pred.shape == label.shape
        error_total = jnp.mean((pred - label) ** 2)

        return error_total

    def _step(state: TrainState, batch: ja) -> Tuple[StepResult, TrainState]:
        loss, grads = jax.value_and_grad(_loss)(state.params, batch)

        grad_norm = global_norm(grads)

        # Gradient step.
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        return StepResult(loss, grad_norm), state._replace(params=params, opt_state=opt_state)

    return _step


def main():
    with open(PKL_PATH, "rb") as f:
        d: TrainData = cloudpickle.load(f)
    batch_size, dt, T, lr, n_hidden = d["batch_size"], d["dt"], d["T"], d["lr"], d["n_hidden"]
    n_samples = d["n_samples"]

    data, test_data = jnp.array(d["train_data"]), jnp.array(d["test_data"])
    n_epochs = d["n_epochs"]

    # ---------------------- Overrides ------------------------
    n_hidden = 32
    dt = 1e-4
    # ---------------------- Overrides ------------------------

    print("data.shape={}, T={}, dt={}, n_hidden={}".format(data.shape, T, dt, n_hidden))

    max_t = T

    dim = int(data.shape[2] // 2)
    rng = hk.PRNGSequence(141)

    model = get_model(n_hidden, dim)
    optimizer = get_optim(lr)
    state = init_state(model, optimizer)

    integrate1 = get_leapfrog(model, dt, max_t, scan=False)
    integrate2 = get_leapfrog(model, dt, max_t, scan=True)

    vmap_integrate1 = jax.vmap(integrate1, in_axes=(None, 0, 0), out_axes=1)
    vmap_integrate2 = jax.vmap(integrate2, in_axes=(None, 0, 0), out_axes=1)

    step = get_step(integrate1, optimizer, dim, max_t)
    step = jax.jit(step)

    idx = 0
    for epoch in track(range(n_epochs)):
        perm = jax.random.permutation(next(rng), n_samples)
        data_perm = data[:, perm]

        for ii in range(0, n_samples, batch_size):
            if ii + batch_size > n_samples:
                break

            data_batch = data_perm[:max_t, ii : ii + batch_size, :]

            # Check difference between the two methods.
            p0, q0 = data_batch[0, :, :dim], data_batch[0, :, dim:]
            # (T, batch, 2 * dim)
            traj1 = vmap_integrate1(state.params, p0, q0)
            traj2 = vmap_integrate2(state.params, p0, q0)
            #
            # diff = jnp.sum((traj1 - traj2) ** 2)
            # traj1, traj2 = traj1[0], traj2[0]

            # # Check the difference between traj and [p0 q0].
            # diff = jnp.sum((traj1[0] - data_batch[0]) ** 2)
            # print("traj0 is x0? diff={:10.4e}".format(diff))

            #
            # print("diff={:10.3e}".format(diff))
            # print(traj1.flatten()[:6 * dim])
            # print(traj2.flatten()[:6 * dim])

            res, state = step(state, data_batch)

            idx += 1

            if idx % 50 == 0:
                print(
                    "idx={:5}, e={:3}, i={:3}: loss={:9.2e}, grad={:9.2e}".format(
                        idx, epoch, ii, res.loss, res.grad_norm
                    )
                )


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
