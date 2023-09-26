import logging
from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import chex
import haiku as hk
import ipdb
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.linalg import expm

from models.input_normalizer import InputNormalizer
from models.mutliso3rnn_util import (
    analytic_dVdq,
    analytic_dVdR,
    analytical_V,
    rigid_correction,
)
from utils.custom_acts import int_gelu
from utils.haiku_utils import (
    MLP,
    ActivatedLinear,
    MLPCfg,
    ParamNames,
    ResnetFC,
    get_mlp,
    scale_init,
    without_apply_rng,
)
from utils.lie import RotAxisAngle, RotZ, hat, vee
from utils.types import ja

# from rich import print


log = logging.getLogger(__file__)


def tostr(arr) -> str:
    s = ["{:12.5e}".format(n) for n in arr.flatten()]
    s = ", ".join(s)
    return "[{}]".format(s)


def make_point_input(q: ja) -> ja:
    n_bodies, _ = q.shape

    # Compute pairwise differences.
    # (n_bodies, 1, 3)
    q1 = q[:, None, :]
    # (1, n_bodies, 3)
    q2 = q[None, :, :]

    # (n_bodies, n_bodies, 3)
    q_diff = q1 - q2
    q_diff = jnp.where(q_diff == 0, 1.0, q_diff)
    # (n_bodies, n_bodies)
    q_norm_sq = jnp.sum(q_diff ** 2, axis=2)

    # Take only the upper triangular elements.
    iu = np.triu_indices(n_bodies, 1)

    # (tri_size, 3)
    triu_q_diff = q_diff[iu]
    flat_q_diff, flat_q_norm_sq = triu_q_diff.flatten(), q_norm_sq[iu]

    # Compute outer norm of q_diff.
    triu_q_diff_vec = jnp.expand_dims(triu_q_diff, axis=2)
    triu_q_diff_vec_T = jnp.expand_dims(triu_q_diff, axis=1)
    # (tri_size, 3, 3)
    q_diff_outer = triu_q_diff_vec * triu_q_diff_vec_T
    flat_q_diff_outer = q_diff_outer.flatten()

    flat_q_norm = jnp.sqrt(flat_q_norm_sq)

    flat_q_norm_3 = flat_q_norm * flat_q_norm_sq
    flat_q_norm_5 = flat_q_norm_3 * flat_q_norm_sq

    tri_size = n_bodies * (n_bodies - 1) // 2

    assert flat_q_diff.shape == (tri_size * 3,)
    assert flat_q_norm_sq.shape == (tri_size,)
    assert flat_q_diff_outer.shape == (tri_size * 9,)

    V_input = jnp.concatenate(
        [flat_q_diff, flat_q_diff_outer, 1 / flat_q_norm, 1 / flat_q_norm_sq, 1 / flat_q_norm_3, 1 / flat_q_norm_5],
        axis=0,
    )

    total_size = tri_size * 16
    assert V_input.shape == (total_size,)

    return V_input


def make_rigid_input(q: ja, R: ja, Js: ja) -> ja:
    n_bodies, _ = q.shape

    R_T = R.swapaxes(-2, -1)
    # RJR = R @ Js @ R_T
    RJR = R @ R_T
    flat_RJR = RJR.reshape(n_bodies * 3 * 3)

    flat_R = R.reshape(n_bodies * 3 * 3)
    rigid_input = jnp.concatenate([flat_R, flat_RJR], axis=0)
    assert rigid_input.ndim == 1

    # # Compute shifted log transform of the rigid input.
    # # fmt: off
    # min_val = jnp.array([
    #     -1.0e+00, -1.0e+00, 1.3e-06, -1.0e+00, -1.0e+00, 1.3e-03, -1.3e-03, -1.3e-03, 1.0e+00,
    #     -1.0e+00, -1.0e+00, -9.9e-01, -1.0e+00, -1.0e+00, -9.8e-01, -1.0e+00, -1.0e+00, -2.4e-01,
    #     1.3e+00, 3.3e-14, 2.5e-11, 3.3e-14, 1.3e+00, 2.5e-08, 2.5e-11, 2.5e-08, 1.3e+00,
    #     1.1e-04, -5.3e-05, -5.0e-06, -5.3e-05, 1.1e-04, -2.3e-05, -5.0e-06, -2.3e-05, 1.1e-04
    # ])
    # # fmt: on
    # log_rigid_input = jnp.log(jnp.abs(rigid_input - min_val + 1e-5))
    #
    # # Normalize so this doesn't destroy the scaling.
    # log_means = jnp.array([
    #     -6.9e-01, -6.8e-01, -1.4e+01, -7.0e-01, -6.9e-01, -1.0e+01, -7.3e+00, -7.3e+00, -1.4e+01,
    #     -2.2e-01, -2.3e-01, -6.4e-01, -2.3e-01, -2.3e-01, -7.1e-01, -5.5e-01, -5.5e-01, -2.0e+00,
    #     -3.5e+00, -1.8e+01, -1.8e+01, -1.8e+01, -3.5e+00, -1.8e+01, -1.8e+01, -1.8e+01, -3.5e+00,
    #     -1.0e+01, -1.0e+01, -1.2e+01, -1.0e+01, -1.0e+01, -1.2e+01, -1.2e+01, -1.2e+01, -1.2e+01
    # ])
    # log_stds = jnp.array([
    #     1.8e+00, 1.8e+00, 1.2e+00, 1.8e+00, 1.8e+00, 2.0e-02, 1.6e+00, 1.6e+00, 1.2e-03,
    #     8.4e-01, 8.8e-01, 1.6e+00, 8.6e-01, 8.6e-01, 1.8e+00, 1.4e+00, 1.4e+00, 1.5e+00,
    #     3.0e-13, 2.0e-06, 1.5e-03, 2.0e-06, 1.4e-12, 1.4e-03, 1.5e-03, 1.4e-03, 2.6e-12,
    #     1.2e+00, 1.5e+00, 1.5e+00, 1.5e+00, 1.2e+00, 1.6e+00, 1.5e+00, 1.6e+00, 5.4e-01
    # ])
    # log_rigid_input = (log_rigid_input - log_means) / log_stds
    #
    # rigid_input = jnp.concatenate([rigid_input, log_rigid_input], axis=0)

    return rigid_input


def make_nn_input(q: ja, R: ja, Js: ja) -> ja:
    point_input = make_point_input(q)
    rigid_input = make_rigid_input(q, R, Js)
    nn_input = jnp.concatenate([point_input, rigid_input], axis=0)

    return nn_input


class AnalyticMode(Enum):
    NoAnalytic = 0
    AnalyticPoint = 1
    AnalyticRigidTrace = 2
    AnalyticRigidFull = 3

    def is_analytic(self) -> bool:
        return self in [AnalyticMode.AnalyticPoint, AnalyticMode.AnalyticRigidTrace, AnalyticMode.AnalyticRigidFull]

    def __bool__(self) -> bool:
        raise RuntimeError("Should not be used as bool!")


class dRPiMode(Enum):
    General = 0
    Analytic = 1

    def __bool__(self) -> bool:
        raise RuntimeError("Should not be used as bool!")


class RigidVNet2(hk.Module):
    def __init__(self, final_weight_coeff: float, name: Optional[str] = None):
        super().__init__(name=name)

        hidden_dim = 128
        n_res_blocks = 3
        act = jax.nn.silu

        self.net = ResnetFC(hidden_dim, n_res_blocks, act, name="enc_fc")
        self.dec = hk.Sequential(
            [
                hk.Linear(hidden_dim),
                act,
                hk.Linear(1, name="last_layer"),
                hk.Linear(1, with_bias=False, w_init=hk.initializers.Constant(final_weight_coeff), name="last_scale"),
            ],
            name="dec",
        )

    def __call__(self, inputs: ja) -> ja:
        out = self.net(inputs)
        return self.dec(out)


class RigidVNet(hk.Module):
    def __init__(
        self,
        w_init_scale: float,
        final_weight_coeff: float,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        act = jax.nn.swish
        # act = int_gelu
        self.act = act

        hidden = 256

        self.R_enc = MLP([hidden, hidden], activation=act, name="R_enc")
        self.RR_enc = MLP([hidden, hidden], activation=act, name="RR_enc")

        self.enc = MLP([hidden], w_init_scale, activation=act, name="enc")
        self.block = MLP([hidden, hidden], w_init_scale, activation=act, name="block")
        self.dec = MLP([hidden, hidden, 1], w_init_scale, activation=act, name="dec")
        scale_init(self.dec.layers[-1].w_init, final_weight_coeff)

        self.norm1 = hk.GroupNorm(32, axis=-1)
        self.norm2 = hk.GroupNorm(32, axis=-1)

    def __call__(self, inputs: jnp.ndarray):
        # inputs: (nx, )
        n_pt = 16
        n_bodies = 2

        RR_start = n_pt + 9 * n_bodies

        pt_input = inputs[:n_pt]
        Rs = inputs[n_pt:RR_start]
        RRs = inputs[RR_start:]

        enc_Rs = self.R_enc(Rs)
        enc_RRs = self.RR_enc(RRs)

        out = jnp.concatenate([pt_input, enc_Rs, enc_RRs], axis=0)

        out = self.enc(out)

        out = self.norm1(out)
        out = self.act(out)

        out = self.block(out)

        out = self.norm2(out)
        out = self.act(out)

        out = self.dec(out)

        return out


class RigidVxNet(hk.Module):
    def __init__(
        self,
        w_init_scale: float,
        final_weight_coeff: float,
        n_bodies: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        act = jax.nn.swish
        self.act = act

        hidden = 256

        self.enc = MLP([hidden], w_init_scale, activation=act, name="enc")
        self.block = MLP([hidden, hidden], w_init_scale, activation=act, name="block")
        self.dec = MLP([hidden, hidden, n_bodies * 12], w_init_scale, activation=act, name="dec")
        scale_init(self.dec.layers[-1].w_init, final_weight_coeff)

        # Output dR (n_bodies * 9) and dq (n_bodies * 3)

        self.norm1 = hk.GroupNorm(32, axis=-1)
        self.norm2 = hk.GroupNorm(32, axis=-1)

    def __call__(self, inputs: jnp.ndarray):
        out = self.enc(inputs)
        out = self.norm1(out)
        out = self.act(out)
        out = self.block(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.dec(out)

        return out


class MultiSO3RNN(hk.Module):
    def __init__(
        self,
        point_V_cfg: MLPCfg,
        rigid_V_cfg: MLPCfg,
        n_bodies: int,
        learn_Vx: bool,
        rigid_V_coeff: float = 1e-15,
        analytic_mode: AnalyticMode = AnalyticMode.NoAnalytic,
        dRPi_mode: dRPiMode = dRPiMode.General,
    ):
        super().__init__()

        # Potential fn.
        point_input_size = n_bodies * 3
        rigid_input_size = n_bodies * 12
        self.point_V_net = get_mlp(1, point_V_cfg, std_init_input_size=point_input_size, name="point_V_net")

        # self.rigid_V_net = get_mlp(1, rigid_V_cfg, std_init_input_size=rigid_input_size, name="rigid_V_net")

        # # || cat(dx, R) - f( features) || ^2, so the NN should output size of n_bodies * 12
        # self.rigid_V_net = get_mlp(41, rigid_V_cfg, std_init_input_size=rigid_input_size, name="rigid_V_net")

        self.rigid_V_net = RigidVNet(rigid_V_cfg.w_init_gain, rigid_V_cfg.final_weight_coeff, name="rigid_V_net")
        # self.rigid_V_net = RigidVNet2(rigid_V_cfg.final_weight_coeff, name="rigid_V_net")

        # self.rigid_Vx_net = RigidVxNet(
        #     rigid_V_cfg.w_init_gain, rigid_V_cfg.final_weight_coeff, n_bodies, name="rigid_Vx_net"
        # )
        # self.learn_Vx = False

        # self.Vq_net = get_mlp(n_bodies * 3, V_cfg, std_init_input_size=rigid_input_size, name="dVdq_net")
        # self.VR_net = get_mlp(n_bodies * 9, V_cfg, std_init_input_size=rigid_input_size, name="dVdR_net")

        self.rigid_V_coeff = rigid_V_coeff
        self.analytic_mode = analytic_mode
        self.dRPi_mode = dRPi_mode

        # If we're using the analytic expression, don't mess with dVdR.
        if self.analytic_mode.is_analytic():
            self.rigid_V_coeff = 1.0

        self.normalizer = InputNormalizer("normalizer")

        # Control NN.
        # self.g_net = get_mlp(self.R_dim, g_cfg, "g_net")

    @staticmethod
    def get_masses(n_bodies: int) -> ja:
        # masses = hk.get_parameter("M", [n_bodies], jnp.float64, init=hk.initializers.RandomNormal())
        # positive_masses = jax.nn.softplus(masses)

        # # TRAPPIST-1
        # positive_masses = jnp.array([0.08, 2.97733e-06])

        # # Sun-Earth-Moon
        # positive_masses = jnp.array([1.0, 3.00329789031572885969e-06, 3.69566879083389693169e-08])

        # Toy
        positive_masses = jnp.array([0.1, 5e-05])

        return positive_masses[:n_bodies]

    @staticmethod
    def _get_diag_Js() -> ja:
        # # TRAPPIST-1
        # inertias = jnp.array(
        #     [
        #         [1.0151095287031408e-08, 1.0151095287031408e-08, 1.015124755460271e-08],
        #         [2.2720513118107325e-15, 2.2720513118107325e-15, 2.2528157259494534e-15],
        #     ]
        # )

        # # Sun-Earth-Moon
        # inertias = jnp.array(
        #     [[8.65073830e-06, 8.65073830e-06, 8.65073830e-06], [2.18128260e-15, 2.18128260e-15, 2.18860213e-15]]
        # )

        # Toy
        inertias = jnp.array(
            [
                [1.268886910878926, 1.268886910878926, 1.268905944325339],
                [0.00011171021771239823, 0.00011171021771239823, 0.00022341852497598791],
            ]
        )

        return inertias

    @staticmethod
    def get_true_Js(n_bodies: int) -> ja:
        inertias = MultiSO3RNN._get_diag_Js()[:n_bodies]

        J = jnp.zeros((n_bodies, 3, 3))
        arange = jnp.arange(3)
        J = J.at[:, arange, arange].set(inertias)

        assert J.shape == (n_bodies, 3, 3)
        return J

    @staticmethod
    def get_true_inv_Js(n_bodies: int) -> ja:
        inertias = MultiSO3RNN._get_diag_Js()[:n_bodies]

        J = jnp.zeros((n_bodies, 3, 3))
        arange = jnp.arange(3)
        J = J.at[:, arange, arange].set(1 / inertias)

        assert J.shape == (n_bodies, 3, 3)
        return J

    def get_Js(self, n_bodies: int) -> ja:
        return self.get_true_Js(n_bodies)

    def get_inv_Js(self, n_bodies: int) -> ja:
        return self.get_true_inv_Js(n_bodies)

    def pred_p_theta(self, d_thetas: ja) -> ja:
        ...

    def pred_d_theta(self, p_thetas: ja) -> ja:
        ...

    def get_V(self, q: ja, R: ja) -> ja:
        if self.analytic_mode.is_analytic():
            rigid_trace = self.analytic_mode in [AnalyticMode.AnalyticRigidTrace, AnalyticMode.AnalyticRigidFull]
            rigid_full = self.analytic_mode == AnalyticMode.AnalyticRigidFull

            dummy = 0.0 * self.nn_V(q, R)
            return self.analytical_V(q, R, rigid_trace, rigid_full) + dummy
        else:
            return self.nn_V(q, R)

    def get_dV_dq(self, q: ja, R: ja) -> ja:
        n_bodies, _ = q.shape

        if self.analytic_mode.is_analytic():
            rigid_trace = self.analytic_mode in [AnalyticMode.AnalyticRigidTrace, AnalyticMode.AnalyticRigidFull]
            rigid_full = self.analytic_mode == AnalyticMode.AnalyticRigidFull

            dV_dq = self.analytic_dVdq(q, R, rigid_trace, rigid_full)
        else:
            dV_dq = hk.grad(self.get_V, argnums=0)(q, R)

            # if self.learn_Vx:
            #     _, nn_input = self.make_nn_input(q, R)
            #     dV_dq = dV_dq + self.rigid_Vx_net(nn_input)[: n_bodies * 3].reshape((n_bodies, 3))

        assert dV_dq.shape == (n_bodies, 3)
        return dV_dq

    def get_dV_dR(self, q: ja, R: ja) -> ja:
        n_bodies, _ = q.shape

        if self.analytic_mode.is_analytic():
            rigid_trace = self.analytic_mode in [AnalyticMode.AnalyticRigidTrace, AnalyticMode.AnalyticRigidFull]
            rigid_full = self.analytic_mode == AnalyticMode.AnalyticRigidFull

            dV_dR = self.analytic_dVdR(q, R, rigid_trace, rigid_full)
        else:
            dV_dR = hk.grad(self.get_V, argnums=1)(q, R)

            # if self.learn_Vx:
            #     _, nn_input = self.make_nn_input(q, R)
            #     dV_dR = dV_dR + self.rigid_Vx_net(nn_input)[n_bodies * 3 :].reshape((n_bodies, 3, 3))

        assert dV_dR.shape == (n_bodies, 3, 3)
        return dV_dR

    def make_nn_input(self, q: ja, R: ja) -> Tuple[ja, ja]:
        n_bodies, _ = q.shape
        Js = self.get_Js(n_bodies)

        point_input = make_point_input(q)
        rigid_input = make_rigid_input(q, R, Js)

        (n_point_inputs,) = point_input.shape

        nn_input = jnp.concatenate([point_input, rigid_input], axis=0)
        nn_input = self.normalizer(nn_input)

        point_input = nn_input[:n_point_inputs]

        return point_input, nn_input

    def nn_V(self, q: ja, R: ja) -> ja:
        # if self.learn_Vx:
        #     self.analytical_V(q, R, rigid_body=False)
        #
        point_input, nn_input = self.make_nn_input(q, R)
        point_V = self.point_V_net(point_input).squeeze()

        # rigid_correction_term = self.rigid_correction(q, R)
        # nn_input = jnp.stack([rigid_correction_term, nn_input[0]])
        # assert nn_input.shape == (2,)

        # rigid_V = rigid_correction_term
        rigid_V = self.rigid_V_coeff * self.rigid_V_net(nn_input).squeeze()
        # rigid_V = 0 * rigid_V + rigid_correction_term
        # rigid_V = self.rigid_V_coeff * jnp.linalg.norm(nn_input - self.rigid_V_net(nn_input)).squeeze()

        log.info("using analytical point_V, rigid_V_coeff={}".format(self.rigid_V_coeff))
        return self.analytical_V(q, R, rigid_trace=False, rigid_full=False) + 0.0 * point_V + rigid_V

        # tmp = 0 * point_V + 0 * rigid_V
        # return self.analytical_V(q, R, rigid_body=True) + tmp

        # print("using point_V, rigid_coeff={}".format(self.rigid_V_coeff))
        # return point_V + self.rigid_V_coeff * rigid_V

    def rigid_V(self, q: ja, R: ja) -> ja:
        # (*, )
        point_input, nn_input = self.make_nn_input(q, R)

        # return self.rigid_V_coeff * jnp.linalg.norm(nn_input - self.rigid_V_net(nn_input)).squeeze()

        rigid_V = self.rigid_V_coeff * self.rigid_V_net(nn_input).squeeze()
        return rigid_V

    def rigid_correction(self, q: ja, R: ja, rigid_trace: bool = True, rigid_full: bool = True) -> ja:
        # q: (n_bodies, 3)
        n_bodies, _ = q.shape

        # (n_bodies, )
        mass = self.get_masses(n_bodies)
        Js = self.get_Js(n_bodies)

        return rigid_correction(q, R, mass, Js, rigid_trace, rigid_full)

    def analytic_dVdq(self, q: ja, R: ja, rigid_trace, rigid_full: bool) -> ja:
        # q: (n_bodies, 3)
        n_bodies, _ = q.shape

        # (n_bodies, )
        mass = self.get_masses(n_bodies)
        # (n_bodies, 3, 3)
        Js = self.get_Js(n_bodies)

        return analytic_dVdq(q, R, mass, Js, rigid_trace, rigid_full)

    def analytic_dVdR(self, q: ja, R: ja, rigid_trace: bool, rigid_body: bool) -> ja:
        # q: (n_bodies, 3)
        n_bodies, _ = q.shape

        # (n_bodies, )
        mass = self.get_masses(n_bodies)
        Js = self.get_Js(n_bodies)

        return analytic_dVdR(q, R, mass, Js, rigid_trace, rigid_body)

    def analytical_V(self, q: ja, R: ja, rigid_trace: bool = False, rigid_full: bool = False) -> ja:
        # q: (n_bodies, 3)
        n_bodies, _ = q.shape

        # (n_bodies, )
        mass = self.get_masses(n_bodies)
        Js = self.get_Js(n_bodies)

        return analytical_V(q, R, mass, Js, rigid_trace, rigid_full)

    def hamiltonian(self, q: ja, p: ja, R: ja, Pi: ja) -> ja:
        """
        :param q: (n_bodies, 3)
        :param p: (n_bodies, 3)
        :param R: (n_bodies, 3, 3)
        :param Pi: (n_bodies, 3)
        :return: Hamiltonian. (,)
        """
        n_bodies, _, _ = R.shape

        Ms = self.get_masses(n_bodies)

        # (n_bodies, )
        lin_KEs = 0.5 * jnp.sum(p ** 2, axis=1) / Ms
        lin_KE = jnp.sum(lin_KEs)

        # (n_bodies, 3, 1)
        Pi_vecs = Pi[:, :, None]
        # (n_bodies, 1, 3)
        Pi_vecs_T = Pi[:, None, :]
        # (n_bodies, 3, 3)
        J_inv = self.get_inv_Js(n_bodies)

        ang_KE = 0.5 * jnp.sum(Pi_vecs_T @ J_inv @ Pi_vecs)

        V = self.get_V(q, R)
        out = lin_KE + ang_KE + V

        assert out.shape == (1,)
        return out

    def d_state(self, q: ja, p: ja, R: ja, Pi: ja) -> Tuple[ja, ja, ja, ja]:
        # q: (n_bodies, 3), p: (n_bodies, 3)
        # Pi: (n_bodies, 3), Pi: (n_bodies, 3, 3)
        n_bodies, _ = p.shape

        Ms = self.get_masses(n_bodies)[:, None]
        assert Ms.shape == (n_bodies, 1)

        J_inv = self.get_inv_Js(n_bodies)
        assert J_inv.shape == (n_bodies, 3, 3)

        dV_dq = self.get_dV_dq(q, R)

        Pi_vec = Pi[:, :, None]

        dq = p / Ms
        dp = -dV_dq
        dR = R @ hat((J_inv @ Pi_vec).squeeze(-1))
        dPi = self.d_Pi(q, p, R, Pi)

        return dq, dp, dR, dPi

    def d_R(self, q: ja, p: ja, R: ja, Pi: ja) -> ja:
        """Rdot = R A.
        :param q: (n_bodies, 3)
        :param p: (n_bodies, 3)
        :param R: (n_bodies, 3, 3)
        :param Pi: (n_bodies, 3)
        :return: The "A" matrix. (n_bodies, 3, 3)
        """
        n_bodies, _ = q.shape
        J_inv = self.get_inv_Js(n_bodies)
        assert J_inv.shape == (n_bodies, 3, 3)

        Pi_vec = Pi[:, :, None]

        d_R = hat((J_inv @ Pi_vec).squeeze(-1))
        assert d_R.shape == (n_bodies, 3, 3)

        return d_R

    def d_Pi(self, q: ja, p: ja, R: ja, Pi: ja) -> ja:
        # Used for Euler / RK4 only.
        # q, p, Pi: (n_bodies, 3)
        # R: (n_bodies, 3, 3)
        n_bodies, _ = q.shape

        Pi_vec = jnp.expand_dims(Pi, axis=-1)

        J = self.get_Js(n_bodies)
        J_inv = jnp.linalg.inv(J)
        Pi_hat = hat(Pi)
        d_Pi1 = (Pi_hat @ (J_inv @ Pi_vec)).squeeze(-1)

        dV_dR = self.get_dV_dR(q, R)
        assert dV_dR.shape == (n_bodies, 3, 3)

        R_T = R.swapaxes(1, 2)
        dV_dR_T = dV_dR.swapaxes(1, 2)

        cross_product_term = R_T @ dV_dR - dV_dR_T @ R
        assert cross_product_term.shape == (n_bodies, 3, 3)

        d_Pi2 = -vee(cross_product_term)

        assert d_Pi1.shape == (n_bodies, 3)
        assert d_Pi2.shape == (n_bodies, 3)

        return d_Pi1 + d_Pi2

    def d_R_Pi1(self, q: ja, p: ja, R: ja, Pi: ja, h: float) -> Tuple[ja, ja]:
        if self.dRPi_mode == dRPiMode.General:
            return self.nn_d_R_Pi1(q, p, R, Pi, h)
        elif self.dRPi_mode == dRPiMode.Analytic:
            return self.analytic_d_R_Pi1(q, p, R, Pi, h)
        else:
            raise RuntimeError("Unknown dRPi_mode {}".format(self.dRPi_mode))

    def nn_d_R_Pi1(self, q: ja, p: ja, R: ja, Pi: ja, h: float) -> Tuple[ja, ja]:
        # q, p, Pi: (n_bodies, 3)
        # R: (n_bodies, 3, 3)
        n_bodies, _ = q.shape

        # Isotropic J => dPi1 = 0.
        d_R = self.d_R(q, p, R, Pi)
        R_expm = jax.vmap(expm)(d_R * h)
        Pi_expm = jnp.tile(jnp.identity(3), (n_bodies, 1, 1))

        return R_expm, Pi_expm

    def analytic_d_R_Pi1(self, q: ja, p: ja, R: ja, Pi: ja, h: float) -> Tuple[ja, ja]:
        # q, p, Pi: (n_bodies, 3)
        # R: (n_bodies, 3, 3)
        n_bodies, _ = q.shape
        assert Pi.shape == (n_bodies, 3)

        # (n_bodies, 3, 3)
        # TODO: Maybe we should only learn the diagonal elements? Or maybe we can do some splitting thing?
        inv_Js = self.get_inv_Js(n_bodies)
        inv_I1, inv_I2, inv_I3 = inv_Js[:, 0, 0], inv_Js[:, 1, 1], inv_Js[:, 2, 2]

        # 1: Compute theta and gamma. (n_bodies, )
        theta = h * (inv_I3 - inv_I2) * Pi[:, 2]
        gamma = h * jnp.linalg.norm(Pi, axis=1) * inv_I2
        assert theta.shape == (n_bodies,)
        assert gamma.shape == (n_bodies,)

        Rz = RotZ(theta)
        assert Rz.shape == (n_bodies, 3, 3)

        Rz_T = Rz.swapaxes(1, 2)

        Rpi = RotAxisAngle(Pi, gamma)
        assert Rpi.shape == (n_bodies, 3, 3)

        Pi_expm = Rz_T
        R_expm = Rpi @ Rz
        #
        # for ii in range(n_bodies):
        #     print(
        #         "    Body {} - theta={:12.4e}, gamma={:12.4e}, Pi={}\n"
        #         "               Rpi={}\n"
        #         "                Rz={}".format(
        #             ii, float(theta[ii]), float(gamma[ii]), tostr(Pi[ii]), tostr(Rpi[ii]), tostr(Rz[ii])
        #         )
        #     )

        return R_expm, Pi_expm

    def d_q(self, q: ja, p: ja, R: ja, Pi: ja) -> ja:
        """
        :param q: (n_bodies, 3)
        :param p: (n_bodies, 3)
        :param R: (n_bodies, 3, 3)
        :param Pi: (n_bodies, 3)
        :return: p/m
        """
        n_bodies, _ = p.shape

        Ms = self.get_masses(n_bodies)[:, None]
        assert Ms.shape == (n_bodies, 1)

        return p / Ms

    def d_p(self, q: ja, p: ja, R: ja, Pi: ja) -> ja:
        """
        :param q: (n_bodies, 3)
        :param p: (n_bodies, 3)
        :param R: (n_bodies, 3, 3)
        :param Pi: (n_bodies, 3)
        :return: dV/dq. (n_bodies, 3)
        """
        n_bodies, _ = q.shape

        dV_dq = self.get_dV_dq(q, R)
        assert dV_dq.shape == (n_bodies, 3)

        # for ii in range(n_bodies):
        #     print("Body {} PVPq: {}".format(ii, dV_dq[ii]))

        return -dV_dq

    def d_Pi1(self, q: ja, p: ja, R: ja, Pi: ja) -> ja:
        """hat{Pi} J^{-1}. Approximation.
        :param q: (n_bodies, 3)
        :param p: (n_bodies, 3)
        :param R: (n_bodies, 3, 3)
        :param Pi: (n_bodies, 3)
        :return: (n_bodies, 3)
        """
        n_bodies, _ = q.shape

        J = self.get_Js(n_bodies)
        J_inv = jnp.linalg.inv(J)
        Pi_hat = hat(Pi)
        d_Pi1 = Pi_hat @ J_inv

        assert d_Pi1.shape == (n_bodies, 3, 3)
        return d_Pi1

    def d_Pi2(self, q: ja, p: ja, R: ja, Pi: ja) -> ja:
        """Pi_dot =
        :param q: (n_bodies, 3)
        :param p: (n_bodies, 3)
        :param R: (n_bodies, 3, 3)
        :param Pi: (n_bodies, 3)
        :return: (n_bodies, 3)
        """
        n_bodies, _ = q.shape

        # Compute 2nd term: -(R^T dV/dR)
        dV_dR = self.get_dV_dR(q, R)
        assert dV_dR.shape == (n_bodies, 3, 3)

        R_T = R.swapaxes(1, 2)
        dV_dR_T = dV_dR.swapaxes(1, 2)

        cross_product_term = R_T @ dV_dR - dV_dR_T @ R
        assert cross_product_term.shape == (n_bodies, 3, 3)

        second_term = -vee(cross_product_term)
        #
        # for ii in range(n_bodies):
        #     print("i={}, pvpr={}, dPi={}".format(ii, dV_dR[ii].flatten(), second_term[ii]))

        # if u is not None:
        #     normalized_u = (u - self.u_mean) / self.u_std
        #     control_term = (self.g_net(flat_R) * normalized_u).squeeze(-1)
        #     # control_term = u[..., 0] / self.M_diag
        #     assert control_term.shape == (batch,)
        #     second_term = second_term + control_term

        assert second_term.shape == (n_bodies, 3)
        return second_term

    @staticmethod
    def adjust_scale(
        params: optax.Params,
        frozen_params: optax.Params,
        param_names_dict: Dict[str, List[ParamNames]],
        grad_coeff_scale: float,
    ) -> Tuple[optax.Params, optax.Params]:
        if grad_coeff_scale == 1.0:
            log.warning("We're trying to adjust_scale by 1?? {}".format(grad_coeff_scale))
            return params, frozen_params
        log.info("Scaling param grad coeffs by {:.3f}!".format(grad_coeff_scale))

        assert len(param_names_dict) > 0

        for module, param_names in param_names_dict.items():
            for param_name in param_names:
                params, frozen_params = ActivatedLinear.adjust_scale(
                    params, frozen_params, param_name, grad_coeff_scale
                )
                log.info(
                    "param_name: {}, scale: {}".format(
                        param_name.bundle_name, frozen_params[param_name.bundle_name][param_name.scale_name]
                    )
                )

        return params, frozen_params


class TransformedSO3RNN(NamedTuple):
    init: Callable[[chex.PRNGKey, ja, ja, ja, ja], Tuple[hk.Params, hk.Params, Dict[str, List[ParamNames]]]]
    rigid_V: Callable[[hk.Params, ja, ja], ja]
    rigid_correction: Callable[[hk.Params, ja, ja], ja]
    d_q: Callable[[hk.Params, ja, ja, ja, ja], ja]
    d_p: Callable[[hk.Params, ja, ja, ja, ja], ja]
    d_R: Callable[[hk.Params, ja, ja, ja, ja], ja]
    d_R_Pi1: Callable[[hk.Params, ja, ja, ja, ja, float], Tuple[ja, ja]]
    d_Pi1: Callable[[hk.Params, ja, ja, ja, ja], ja]
    d_Pi2: Callable[[hk.Params, ja, ja, ja, ja], ja]
    # Used for Euler / RK4.
    d_state: Callable[[hk.Params, ja, ja, ja, ja], Tuple[ja, ja, ja, ja]]
    param_names: Dict[str, List[ParamNames]]


def get_so3rnn_model(
    point_V_cfg: MLPCfg,
    rigid_V_cfg: MLPCfg,
    n_bodies: int,
    learn_Vx: bool,
    rigid_V_coeff: float = 1.0,
    analytic_mode: AnalyticMode = AnalyticMode.NoAnalytic,
    dRPi_mode: dRPiMode = dRPiMode.General,
) -> Tuple[TransformedSO3RNN, Any]:
    print("analytic_mode={}, dRPi_mode={}, rigid_V_coeff={}".format(analytic_mode, dRPi_mode, rigid_V_coeff))
    log.info("learn_Vx: {}".format(learn_Vx))

    def make_model() -> MultiSO3RNN:
        return MultiSO3RNN(
            point_V_cfg,
            rigid_V_cfg,
            n_bodies,
            learn_Vx,
            rigid_V_coeff=rigid_V_coeff,
            analytic_mode=analytic_mode,
            dRPi_mode=dRPi_mode,
        )

    def _model():
        net = make_model()

        def init(q: ja, p: ja, R: ja, Pi: ja):
            assert R.ndim == 3 and Pi.ndim == 2
            net.d_q(q, p, R, Pi)
            net.d_p(q, p, R, Pi)
            net.d_R(q, p, R, Pi)
            net.d_Pi1(q, p, R, Pi)
            net.d_Pi2(q, p, R, Pi)
            net.nn_V(q, R)

        return init, (
            net.rigid_V,
            net.rigid_correction,
            net.d_q,
            net.d_p,
            net.d_R,
            net.d_R_Pi1,
            net.d_Pi1,
            net.d_Pi2,
            net.d_state,
        )

    model = without_apply_rng(hk.multi_transform(_model))
    rigid_V, _rigid_correction, d_q, d_p, d_R, d_R_Pi1, d_Pi1, d_Pi2, d_state = model.apply

    def _init(
        rng: chex.PRNGKey, q: ja, p: ja, R: ja, Pi: ja
    ) -> Tuple[hk.Params, hk.Params, Dict[str, List[ParamNames]]]:
        params = model.init(rng, q, p, R, Pi)

        return params, {}, {}

    return (
        TransformedSO3RNN(_init, rigid_V, _rigid_correction, d_q, d_p, d_R, d_R_Pi1, d_Pi1, d_Pi2, d_state, {}),
        make_model,
    )
