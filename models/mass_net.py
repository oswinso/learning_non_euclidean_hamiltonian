import haiku as hk

from utils.haiku_utils import MLPCfg, get_mlp
from utils.types import ja
import numpy as np


class MassNet(hk.Module):
    """Output PSD mass matrices"""

    def __init__(self, n_bodies: int, cfg: MLPCfg):
        super().__init__()

        # Number of elements in the 3x3 mass matrix.
        self.total_numels = n_bodies * 9
        # Number of elements in the lower diagonal.
        self.n_tril = n_bodies * 6

        # Only predict the lower triangular matrix, then M = L L^T.
        self.mlp = get_mlp(self.n_tril, cfg, "mass_mlp")

        # diag_idx: diagonals for the lower triangular matrix.
        diag_idx = np.arange(3)
        self.diag_idx = (diag_idx * (diag_idx + 1)) // 2 - 1

        # tri_idx: off-diagonals for the lower triangular matrix.
        self.tri_idx = np.extract([x not in diag_idx for x in np.arange(self.n_tril)], np.arange(self.n_tril))

        self.tril_idx = np.tril_indices(3, 3)


    def __call__(self) -> ja:
        diag_eps = 1e-5
