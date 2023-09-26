import logging
from typing import List

import numpy as np
from attr import s

log = logging.getLogger(__file__)


@s(auto_attribs=True, slots=True, auto_exc=True, auto_detect=True, order=False)
class PlanetsData:
    names: List[str]
    dt: float
    # (nt, n_planets, 3)
    pos: np.ndarray
    # (nt, n_planets, 3)
    vel: np.ndarray
    # (nt, n_planets, 3, 3)
    rot: np.ndarray
    # (nt, n_planets, 3)
    lin_mom: np.ndarray
    # (nt, n_planets, 3)
    ang_mom: np.ndarray

    # (n_planets, )
    mass: np.ndarray
    # (n_planets, 3)
    inertia: np.ndarray

    @property
    def n_bodies(self) -> int:
        return len(self.names)

    @property
    def n_times(self) -> int:
        return self.pos.shape[0]

    @property
    def n_steps(self) -> int:
        return self.n_times - 1
