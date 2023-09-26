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
from integrators.multi_symplectic import IntegrateFn, IntResult, SO3IntState
from utils.expm import expm as my_expm
from utils.jax_utils import concat_at_end, serial_scan
from utils.types import ja

if TYPE_CHECKING:
    from models.multiso3rnn import TransformedSO3RNN


class _OuterState(NamedTuple):
    q: ja
    p: ja
    R: ja
    Pi: ja


def get_multi_euler(
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

    def get_outer_loop(params: hk.Params):
        def _outer_loop(carry: _OuterState, _) -> Tuple:
            dq, dp, dR, dPi = model.d_state(params, carry.q, carry.p, carry.R, carry.Pi)

            assert carry.q.shape == dq.shape
            assert carry.p.shape == dp.shape
            assert carry.R.shape == dR.shape
            assert carry.Pi.shape == dPi.shape

            # Full step.
            q = carry.q + substep_h * dq
            p = carry.p + substep_h * dp
            R = carry.R + substep_h * dR
            Pi = carry.Pi + substep_h * dPi

            return _OuterState(q, p, R, Pi), SO3IntState(carry.q, carry.p, carry.R, carry.Pi)

        return _outer_loop

    def euler(q0: ja, p0: ja, R0: ja, Pi0: ja, params: hk.Params) -> IntResult:
        n_body, _ = q0.shape
        assert q0.shape == (n_body, 3)
        assert p0.shape == (n_body, 3)
        assert R0.shape == (n_body, 3, 3)
        assert Pi0.shape == (n_body, 3)

        outer_loop = get_outer_loop(params)

        if use_serial_scan:
            scan = serial_scan
        else:
            scan = jl.scan

        final_carry, outputs = scan(
            outer_loop,
            _OuterState(q0, p0, R0, Pi0),
            None,
            length=n_steps,
            unroll=unroll,
        )

        qs = concat_at_end(outputs.q, final_carry.q, axis=0)
        ps = concat_at_end(outputs.p, final_carry.p, axis=0)
        Rs = concat_at_end(outputs.R, final_carry.R, axis=0)
        Pis = concat_at_end(outputs.Pi, final_carry.Pi, axis=0)

        return IntResult(qs, ps, Rs, Pis)

    if vmap:

        def _euler(q0: ja, p0: ja, R0: ja, Pi0: ja, params: hk.Params) -> IntResult:
            batch, n_body, three = q0.shape
            if batch == 1:
                q0 = q0.squeeze(0)
                p0 = p0.squeeze(0)
                R0 = R0.squeeze(0)
                Pi0 = Pi0.squeeze(0)
                result = euler(q0, p0, R0, Pi0, params)
                return jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), result)
            else:
                return jax.vmap(euler, in_axes=(0, 0, 0, 0, None))(q0, p0, R0, Pi0, params)

    else:
        _euler = euler

    return _euler
