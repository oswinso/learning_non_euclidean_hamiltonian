from abc import ABC, abstractmethod
from typing import Callable, NamedTuple, Optional, Union

import haiku as hk

from utils.types import ja


class DynParams(NamedTuple):
    ...


class Dynamics(NamedTuple):
    hamiltonian: Callable[[ja, ja, DynParams], ja]
    d_R: Callable[[ja, ja, DynParams], ja]
    d_Pi: Callable[[ja, ja, DynParams, Optional[ja]], ja]
    d_theta: Callable[[ja, DynParams], ja]


class IntegrableDynamics(NamedTuple):
    d_R: Callable[[hk.Params, ja, ja], ja]
    d_Pi: Callable[[hk.Params, ja, ja, Optional[ja]], ja]


class BaseDynamics(ABC):
    @staticmethod
    @abstractmethod
    def hamiltonian(Rs: ja, p_thetas: ja, p: DynParams) -> ja:
        ...

    @staticmethod
    @abstractmethod
    def d_R(R: ja, p_theta: ja, p: DynParams) -> ja:
        ...

    @staticmethod
    @abstractmethod
    def d_Pi(R: ja, p_theta: ja, p: DynParams, u: Optional[ja] = None) -> ja:
        ...

    @staticmethod
    @abstractmethod
    def d_theta(p_theta: ja, p: DynParams) -> ja:
        ...


def create_dynamics(cls: BaseDynamics) -> Dynamics:
    return Dynamics(cls.hamiltonian, cls.d_R, cls.d_Pi, cls.d_theta)


def create_integrable(dyn: Union[Dynamics, BaseDynamics], params: DynParams) -> IntegrableDynamics:
    if not isinstance(dyn, Dynamics):
        dyn = create_dynamics(dyn)

    def d_R(_: hk.Params, R: ja, p_theta: ja) -> ja:
        return dyn.d_R(R, p_theta, params)

    def d_Pi(_: hk.Params, R: ja, p_theta: ja, u: Optional[ja] = None) -> ja:
        return dyn.d_Pi(R, p_theta, params, u)

    return IntegrableDynamics(d_R, d_Pi)
