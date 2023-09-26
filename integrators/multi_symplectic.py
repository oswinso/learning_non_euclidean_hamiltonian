from functools import partial
from typing import TYPE_CHECKING, NamedTuple, Protocol, Tuple

import haiku as hk
import jax
import jax.lax as jl
import jax.numpy as jnp
from jax.experimental import loops
from jax.scipy.linalg import expm

from dynamics.base_dynamics import BaseDynamics
from dynamics.base_integrable import IntegrableProtocol, MultiIntegrableProtocol
from utils.expm import expm as my_expm
from utils.jax_utils import concat_at_end, serial_scan
from utils.types import ja

if TYPE_CHECKING:
    from models.multiso3rnn import TransformedSO3RNN


class IntResult(NamedTuple):
    qs: ja
    ps: ja
    Rs: ja
    Pis: ja


class SO3IntOuterState(NamedTuple):
    q: ja
    p: ja
    R: ja
    Pi: ja
    # expm(R_dot * substep_h / 2)
    R_half_expm: ja
    # expm(Pi_dot * substep_h / 2)
    Pi_half_expm: ja


class SO3IntState(NamedTuple):
    q: ja
    p: ja
    R: ja
    Pi: ja


class IntegrateFn(Protocol):
    def __call__(self, q0s: ja, p0s: ja, R0s: ja, Pi0s: ja, params: hk.Params) -> IntResult:
        ...


def get_multi_verlet(
    model: "TransformedSO3RNN",
    h: float,
    n_steps: int,
    n_substeps: int = 1,
    unroll: int = 8,
    vmap: bool = True,
    use_serial_scan: bool = False,
) -> IntegrateFn:
    assert n_substeps > 0
    substep_h = h / n_substeps
    half_substep = substep_h / 2

    # vmap_expm = jax.vmap(partial(expm, max_squarings=16))
    # vmap_expm = jax.vmap(partial(my_expm, max_squarings=16))

    def get_outer_loop(params: hk.Params):
        # def _inner_loop(carry: SO3IntState, _) -> Tuple[SO3IntState, None]:
        #     # # AB: A full step.
        #     # R_full_expm = vmap_expm(model.d_R(params, carry.q, carry.p, carry.R, carry.Pi) * substep_h)
        #     # Pi_full_expm = vmap_expm(model.d_Pi1(params, carry.q, carry.p, carry.R, carry.Pi) * substep_h)
        #
        #     R_full_expm, Pi_full_expm = model.d_R_Pi1(params, carry.q, carry.p, carry.R, carry.Pi, substep_h)
        #
        #     A_q = carry.q + substep_h * model.d_q(params, carry.q, carry.p, carry.R, carry.Pi)
        #     A_p = carry.p
        #     A_R = carry.R @ R_full_expm
        #     A_Pi = jnp.squeeze(Pi_full_expm @ carry.Pi[:, :, None], axis=-1)
        #     # A_Pi = carry.Pi
        #
        #     # AB: B full step.
        #     B_q = A_q
        #     B_p = A_p + substep_h * model.d_p(params, A_q, A_p, A_R, A_Pi)
        #     B_R = A_R
        #     B_Pi = A_Pi + substep_h * model.d_Pi2(params, A_q, A_p, A_R, A_Pi)
        #
        #     return SO3IntState(B_q, B_p, B_R, B_Pi), None
        #

        def tostr(arr) -> str:
            s = ["{:12.5e}".format(n) for n in arr.flatten()]
            s = ", ".join(s)
            return "[{}]".format(s)

        def _outer_loop(carry: SO3IntOuterState, _) -> Tuple[SO3IntOuterState, SO3IntState]:
            # print("\n------------------------- current_time=??  -----------------")
            # for ii in range(2):
            #     print("body {}".format(ii))
            #     print("    q:{}, p:{}".format(tostr(carry.q[ii]), tostr(carry.p[ii])))
            #     print("    R:{}".format(tostr(carry.R[ii])))
            #     print("   Pi:{}".format(tostr(carry.Pi[ii])))
            #
            # RECOMPUTE R_half_expm and Pi_half_expm since GRIT doesn't cache these matrices and actually recomputes
            # these at the end / beginning of each output using the NEW Pi.
            R_half_expm, Pi_half_expm = model.d_R_Pi1(params, carry.q, carry.p, carry.R, carry.Pi, half_substep)

            # AB: A half step.
            A_q = carry.q + half_substep * model.d_q(params, carry.q, carry.p, carry.R, carry.Pi)
            A_p = carry.p
            A_R = carry.R @ R_half_expm
            A_Pi = jnp.squeeze(Pi_half_expm @ carry.Pi[:, :, None], axis=-1)
            # A_Pi = carry.Pi

            # AB: B full step.
            B_q = A_q
            B_p = A_p + substep_h * model.d_p(params, A_q, A_p, A_R, A_Pi)
            B_R = A_R
            B_Pi = A_Pi + substep_h * model.d_Pi2(params, A_q, A_p, A_R, A_Pi)

            # if n_substeps > 1:
            #     final_carry, _ = jl.scan(_inner_loop, SO3IntState(B_q, B_p, B_R, B_Pi), None, length=n_substeps - 1)
            #
            #     B_q, B_p, B_R, B_Pi = final_carry.q, final_carry.p, final_carry.R, final_carry.Pi

            # AB: A half step.
            # d_R = model.d_R(params, B_q, B_p, B_R, B_Pi)
            # R_half_expm = vmap_expm(d_R * half_substep)
            # d_Pi1 = model.d_Pi1(params, B_q, B_p, B_R, B_Pi)
            # Pi_half_expm = vmap_expm(d_Pi1 * half_substep)
            R_half_expm, Pi_half_expm = model.d_R_Pi1(params, B_q, B_p, B_R, B_Pi, half_substep)

            new_A_q = B_q + half_substep * model.d_q(params, B_q, B_p, B_R, B_Pi)
            new_A_p = B_p
            new_A_R = B_R @ R_half_expm
            new_A_Pi = jnp.squeeze(Pi_half_expm @ B_Pi[:, :, None], axis=-1)
            # new_A_Pi = B_Pi

            return SO3IntOuterState(new_A_q, new_A_p, new_A_R, new_A_Pi, R_half_expm, Pi_half_expm), SO3IntState(
                carry.q, carry.p, carry.R, carry.Pi
            )

        return _outer_loop

    def verlet(q0: ja, p0: ja, R0: ja, Pi0: ja, params: hk.Params) -> IntResult:
        n_body, _ = q0.shape
        assert q0.shape == (n_body, 3)
        assert p0.shape == (n_body, 3)
        assert R0.shape == (n_body, 3, 3)
        assert Pi0.shape == (n_body, 3)

        outer_loop = get_outer_loop(params)

        # # R_half_expm = vmap_expm(model.d_R(params, q0, p0, R0, Pi0) * half_substep)
        # # Pi_half_expm = vmap_expm(model.d_Pi1(params, q0, p0, R0, Pi0) * half_substep)
        R_half_expm, Pi_half_expm = model.d_R_Pi1(params, q0, p0, R0, Pi0, half_substep)

        if use_serial_scan:
            scan = serial_scan
        else:
            scan = jl.scan

        final_carry, outputs = scan(
            outer_loop,
            SO3IntOuterState(q0, p0, R0, Pi0, R_half_expm, Pi_half_expm),
            None,
            length=n_steps,
            unroll=unroll,
        )

        qs = concat_at_end(outputs.q, final_carry.q, axis=0)
        ps = concat_at_end(outputs.p, final_carry.p, axis=0)
        Rs = concat_at_end(outputs.R, final_carry.R, axis=0)
        Pis = concat_at_end(outputs.Pi, final_carry.Pi, axis=0)

        return IntResult(qs, ps, Rs, Pis)

    # Don't vmap over params.
    if vmap:

        def _verlet(q0: ja, p0: ja, R0: ja, Pi0: ja, params: hk.Params) -> IntResult:
            batch, n_body, three = q0.shape
            if batch == 1:
                q0 = q0.squeeze(0)
                p0 = p0.squeeze(0)
                R0 = R0.squeeze(0)
                Pi0 = Pi0.squeeze(0)
                result = verlet(q0, p0, R0, Pi0, params)
                return jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), result)
            else:
                return jax.vmap(verlet, in_axes=(0, 0, 0, 0, None))(q0, p0, R0, Pi0, params)

    else:
        _verlet = verlet

    return _verlet


def get_multi_verlet_norot(
    model: "TransformedSO3RNN", h: float, n_steps: int, n_substeps: int = 1, unroll: int = 8, vmap: bool = True
) -> IntegrateFn:
    assert n_substeps > 0
    substep_h = h / n_substeps
    half_substep = substep_h / 2

    zero = jnp.zeros(1)

    def get_outer_loop(params: hk.Params):
        def _outer_loop(carry: SO3IntOuterState, _) -> Tuple[SO3IntOuterState, SO3IntState]:
            A_q = carry.q + half_substep * model.d_q(params, carry.q, carry.p, carry.R, carry.Pi)
            A_p = carry.p

            # AB: B full step.
            d_p = model.d_p(params, A_q, A_p, carry.R, carry.Pi)
            B_q = A_q
            B_p = A_p + substep_h * d_p

            # AB: A half step.
            new_A_q = B_q + half_substep * model.d_q(params, B_q, B_p, carry.R, carry.Pi)
            new_A_p = B_p

            return SO3IntOuterState(new_A_q, new_A_p, carry.R, carry.Pi, zero, zero), SO3IntState(
                carry.q, carry.p, carry.R, carry.Pi
            )

        return _outer_loop

    def verlet(q0: ja, p0: ja, R0: ja, Pi0: ja, params: hk.Params) -> IntResult:
        n_body, _ = q0.shape
        assert q0.shape == (n_body, 3)
        assert p0.shape == (n_body, 3)
        assert R0.shape == (n_body, 3, 3)
        assert Pi0.shape == (n_body, 3)

        outer_loop = get_outer_loop(params)

        final_carry, outputs = jl.scan(
            outer_loop, SO3IntOuterState(q0, p0, R0, Pi0, zero, zero), None, length=n_steps, unroll=unroll
        )

        qs = concat_at_end(outputs.q, final_carry.q, axis=0)
        ps = concat_at_end(outputs.p, final_carry.p, axis=0)
        Rs = concat_at_end(outputs.R, final_carry.R, axis=0)
        Pis = concat_at_end(outputs.Pi, final_carry.Pi, axis=0)

        return IntResult(qs, ps, Rs, Pis)

    # Don't vmap over params.
    if vmap:
        verlet = jax.vmap(verlet, in_axes=(0, 0, 0, 0, None))

    def _verlet_wrapper(q0: ja, p0: ja, R0: ja, Pi0: ja, params: hk.Params) -> IntResult:
        batch, n_body, three = q0.shape
        if batch == 1:
            q0 = q0.squeeze(0)
            p0 = p0.squeeze(0)
            R0 = R0.squeeze(0)
            Pi0 = Pi0.squeeze(0)
            result = verlet(q0, p0, R0, Pi0, params)
            return jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), result)
        else:
            return jax.vmap(verlet, in_axes=(0, 0, 0, 0, None))(q0, p0, R0, Pi0, params)

    return _verlet_wrapper
