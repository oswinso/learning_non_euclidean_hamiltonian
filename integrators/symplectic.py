from typing import NamedTuple, Protocol, Tuple

import haiku as hk
import jax
import jax.lax as jl
import jax.numpy as jnp
from jax.experimental import loops
from jax.scipy.linalg import expm

from dynamics.base_dynamics import BaseDynamics
from dynamics.base_integrable import IntegrableProtocol
from models.so2rnn import TransformedSO2RNN
from utils.jax_utils import concat_at_end
from utils.types import ja


class IntResult(NamedTuple):
    Rs: ja
    p_thetas: ja


class SO2IntOuterState(NamedTuple):
    R: ja
    p_theta: ja
    # expm(R_dot * substep_h / 2)
    half_expm_Rdot: ja


class SO2IntState(NamedTuple):
    R: ja
    p_theta: ja


class IntegrateFn(Protocol):
    def __call__(self, R0s: ja, p_theta0s: ja, params: hk.Params) -> IntResult:
        ...


def get_verlet(model: IntegrableProtocol, h: float, n_steps: int, n_substeps: int = 1) -> IntegrateFn:
    assert n_substeps > 0
    substep_h = h / n_substeps

    def get_outer_loop(params: hk.Params):
        def _inner_loop(carry: SO2IntState, u: ja) -> Tuple[SO2IntState, None]:
            full_half_expm = expm(model.d_R(params, carry.R, carry.p_theta) * substep_h)

            R = carry.R @ full_half_expm
            p_theta = carry.p_theta + substep_h * model.d_Pi(params, R, carry.p_theta, u)

            return SO2IntState(R, p_theta), None

        def _outer_loop(carry: SO2IntOuterState, us: ja) -> Tuple[SO2IntOuterState, SO2IntState]:
            R = carry.R @ carry.half_expm_Rdot
            p_theta = carry.p_theta + substep_h * model.d_Pi(params, R, carry.p_theta, us[0])

            if n_substeps > 1:
                final_carry, _ = jl.scan(_inner_loop, SO2IntState(R, p_theta), us[1:])
                p_theta = final_carry.p_theta

            half_expm = expm(model.d_R(params, R, p_theta) * substep_h / 2)
            R_half = R @ half_expm

            return SO2IntOuterState(R_half, p_theta, half_expm), SO2IntState(carry.R, carry.p_theta)

        return _outer_loop

    def verlet(R0: ja, p_theta0: ja, params: hk.Params) -> IntResult:
        us = jnp.zeros((n_steps, 1, 1))
        # (n_steps, n_substeps, nu)
        us = jnp.tile(us, (1, n_substeps, 1))
        assert us.shape == (n_steps, n_substeps, 1)

        outer_loop = get_outer_loop(params)

        half_expm_Rdot = expm(model.d_R(params, R0, p_theta0) * substep_h / 2)
        final_carry, outputs = jl.scan(outer_loop, SO2IntOuterState(R0, p_theta0, half_expm_Rdot), us)

        Rs = concat_at_end(outputs.R, final_carry.R, axis=0)
        p_thetas = concat_at_end(outputs.p_theta, final_carry.p_theta, axis=0)

        return IntResult(Rs, p_thetas)

    # Don't vmap over params.
    vmap_verlet = jax.vmap(verlet, in_axes=(0, 0, None))
    return vmap_verlet


def test_verlet() -> None:
    def d_R(params: hk.Params, R: ja, Pi: ja) -> ja:
        return jnp.log(1) * jnp.eye(2)

    def d_Pi(params: hk.Params, R: ja, Pi: ja, u: ja) -> ja:
        return jnp.ones(1)

    model = TransformedSO2RNN(None, None, None, None, d_R, d_Pi)
    h, n_steps, n_substeps = 1.0, 5, 3
    verlet = get_verlet(model, h, n_steps, n_substeps)

    R0 = jnp.eye(2)
    p_theta0 = jnp.zeros(2)
    result = verlet(R0, p_theta0, params={})

    print(result.Rs)
    print(result.p_thetas)


if __name__ == "__main__":
    import ipdb

    with ipdb.launch_ipdb_on_exception():
        test_verlet()
