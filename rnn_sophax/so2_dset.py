from attr import s

from integrators.symplectic import IntResult


@s(auto_attribs=True, slots=True, auto_exc=True, auto_detect=True, order=False)
class PendDSet:
    trajs: IntResult
    h: float
    n_steps: int
    n_substeps: int
