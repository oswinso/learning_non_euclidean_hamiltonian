from typing import Callable, NamedTuple, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from utils.haiku_utils import MLPCfg, get_mlp, without_apply_rng
from utils.jax_utils import scalarize
from utils.lie import hat_so2, vee_so2
from utils.types import ja


class SO2RNN(hk.Module):
    def __init__(self, V_cfg: MLPCfg, g_cfg: MLPCfg):
        super().__init__()

        self.R_dim = 2 ** 2
        self.w_dim = 1

        # Potential fn.
        self.V_net = get_mlp(1, V_cfg, "V_net")

        # Control NN.
        # self.g_net = get_mlp(self.R_dim, g_cfg, "g_net")

    def get_M_diag(self) -> ja:
        return hk.get_parameter("M_diag", [1], jnp.float64, init=jnp.ones)

    def pred_p_theta(self, d_thetas: ja) -> ja:
        M_diag = self.get_M_diag()
        return M_diag * d_thetas

    def pred_d_theta(self, p_thetas: ja) -> ja:
        return p_thetas / self.get_M_diag()

    def hamiltonian(self, R: ja, Pi: ja) -> ja:
        """
        :param R: (2, 2)
        :param Pi: (,)
        :return: Hamiltonian. (,)
        """
        flat_R = R.reshape(4)
        out = 0.5 * Pi ** 2 / self.get_M_diag() + self.V_net(flat_R)

        assert out.shape == (1,)
        return out

    def d_R(self, R: ja, Pi: ja) -> ja:
        """Rdot = A R.
        :param R: (2, 2)
        :param Pi: (,)
        :return: The "A" matrix. (2, 2)
        """
        M_diag = self.get_M_diag().squeeze()
        return hat_so2(Pi / M_diag)

    def d_Pi(self, R: ja, Pi: ja, u: Optional[ja] = None) -> ja:
        """Pi_dot =
        :param R: (2, 2)
        :param Pi: (1,)
        :param u: (1,)
        :return: (, )
        """
        flat_R = R.reshape(4)
        dV_dR = jax.grad(scalarize(self.V_net))(flat_R)
        assert dV_dR.shape == (4,)
        dV_dR = dV_dR.reshape(2, 2)

        cross_product_term = R.transpose() @ dV_dR - dV_dR.transpose() @ R
        assert cross_product_term.shape == (2, 2)

        second_term = -vee_so2(cross_product_term)

        # if u is not None:
        #     normalized_u = (u - self.u_mean) / self.u_std
        #     control_term = (self.g_net(flat_R) * normalized_u).squeeze(-1)
        #     # control_term = u[..., 0] / self.M_diag
        #     assert control_term.shape == (batch,)
        #     second_term = second_term + control_term

        assert second_term.shape == tuple()
        return second_term


class TransformedSO2RNN(NamedTuple):
    init: Callable[[chex.PRNGKey, ja, ja], ja]
    pred_p_theta: Callable[[hk.Params, ja], ja]
    pred_d_theta: Callable[[hk.Params, ja], ja]
    hamiltonian: Callable[[hk.Params, ja, ja], ja]
    d_R: Callable[[hk.Params, ja, ja], ja]
    d_Pi: Callable[[hk.Params, ja, ja, Optional[ja]], ja]


def get_so2rnn_model(V_cfg: MLPCfg, g_cfg: MLPCfg) -> TransformedSO2RNN:
    def _model():
        net = SO2RNN(V_cfg, g_cfg)

        def init(R: ja, Pi: ja):
            assert R.ndim == 2 and Pi.ndim == 1
            net.d_R(R, Pi)
            return net.d_Pi(R, Pi, None)

        return init, (net.pred_p_theta, net.pred_d_theta, net.hamiltonian, net.d_R, net.d_Pi)

    model = without_apply_rng(hk.multi_transform(_model))
    pred_p_theta, pred_d_theta, hamiltonian, d_R, d_Pi = model.apply

    return TransformedSO2RNN(model.init, pred_p_theta, pred_d_theta, hamiltonian, d_R, d_Pi)
