from typing import List, Optional, Tuple

import numpy as np
import optax
from attr import s

from rnn_sophax import weighted_loss as wl
from utils.types import ja


@s(auto_attribs=True, slots=True, auto_exc=True, auto_detect=True, order=False)
class Phase:
    n_steps: int
    # Lr to apply before gradient clipping.
    preclip_lr: float
    lr_sched: optax.Schedule
    wl_sched: wl.LossWeightSchedule
    # How much to scale the grad coeffs by when the phase starts.
    grad_coeff: Optional[float] = None


def get_warmup_steps(phase_steps: int, warmup_frac: Optional[float], n_warmup_steps: Optional[int]) -> int:
    if n_warmup_steps is None and warmup_frac is None:
        return 0
    elif n_warmup_steps is None and warmup_frac is not None:
        return int(warmup_frac * phase_steps)
    else:
        return n_warmup_steps


def warmup_cosine_decay_flat_schedule(
    init_value: float, peak_value: float, warmup_steps: int, decay_steps: int, end_value: float = 0.0
) -> optax.Schedule:
    schedules = [
        optax.linear_schedule(init_value=init_value, end_value=peak_value, transition_steps=warmup_steps),
        optax.cosine_decay_schedule(
            init_value=peak_value, decay_steps=decay_steps - warmup_steps, alpha=end_value / peak_value
        ),
        optax.constant_schedule(end_value),
    ]
    return optax.join_schedules(schedules, [warmup_steps, decay_steps])


def truncated_flat_phase(
    phase_steps: int,
    final_lr_steps: int,
    trunc_steps: int,
    T: int,
    preclip_lr: float,
    max_lr: float,
    n_warmup_steps: Optional[float] = None,
    warmup_frac: Optional[float] = None,
    final_lr_frac: float = 0.0,
    grad_coeff: Optional[float] = None,
) -> Phase:
    n_warmup_steps = get_warmup_steps(phase_steps, warmup_frac, n_warmup_steps)
    final_lr = final_lr_frac * max_lr

    return Phase(
        n_steps=phase_steps + final_lr_steps,
        preclip_lr=preclip_lr,
        lr_sched=warmup_cosine_decay_flat_schedule(1e-9 * max_lr, max_lr, n_warmup_steps, phase_steps, final_lr),
        wl_sched=wl.truncated_loss(trunc_steps, T),
        grad_coeff=grad_coeff,
    )


def warmup_linear(init_value: float, peak_value: float, warmup_steps: int) -> optax.Schedule:
    schedules = [
        optax.linear_schedule(init_value=init_value, end_value=peak_value, transition_steps=warmup_steps),
        optax.constant_schedule(peak_value),
    ]
    return optax.join_schedules(schedules, [warmup_steps])


def trunc_lin_phase(
    phase_steps: int,
    trunc_steps: int,
    T: int,
    preclip_lr: float,
    max_lr: float,
    n_warmup_steps: Optional[float] = None,
    warmup_frac: Optional[float] = None,
) -> Phase:
    n_warmup_steps = get_warmup_steps(phase_steps, warmup_frac, n_warmup_steps)
    return Phase(
        n_steps=phase_steps,
        preclip_lr=preclip_lr,
        lr_sched=warmup_linear(1e-12 * max_lr, max_lr, n_warmup_steps),
        wl_sched=wl.truncated_loss(trunc_steps, T),
    )


def truncated_phase_onecycle(
    phase_steps: int,
    trunc_steps: int,
    T: int,
    preclip_lr: float,
    max_lr: float,
    warmup_frac: Optional[float] = None,
    final_lr_frac: Optional[float] = None,
) -> Phase:
    if warmup_frac is None:
        warmup_frac = 0.3

    if final_lr_frac is None:
        final_lr_frac = 1e-4

    return Phase(
        n_steps=phase_steps,
        preclip_lr=preclip_lr,
        lr_sched=optax.cosine_onecycle_schedule(phase_steps, max_lr, warmup_frac, final_div_factor=1 / final_lr_frac),
        wl_sched=wl.truncated_loss(trunc_steps, T),
    )


def truncated_phase(
    phase_steps: int,
    trunc_steps: int,
    T: int,
    preclip_lr: float,
    max_lr: float,
    n_warmup_steps: Optional[float] = None,
    warmup_frac: Optional[float] = None,
    final_lr_frac: float = 0.0,
    grad_coeff: Optional[float] = None,
) -> Phase:
    n_warmup_steps = get_warmup_steps(phase_steps, warmup_frac, n_warmup_steps)
    final_lr = final_lr_frac * max_lr

    return Phase(
        n_steps=phase_steps,
        preclip_lr=preclip_lr,
        lr_sched=optax.warmup_cosine_decay_schedule(1e-9 * max_lr, max_lr, n_warmup_steps, phase_steps, final_lr),
        wl_sched=wl.truncated_loss(trunc_steps, T),
        grad_coeff=grad_coeff,
    )


def truncated_transition_phase(
    phase_steps: int,
    prev_trunc_steps: int,
    new_trunc_steps: int,
    max_power: float,
    T: int,
    preclip_lr: float,
    max_lr: float,
    warmup_frac: Optional[float] = None,
    n_warmup_steps: Optional[int] = None,
    final_lr_frac: float = 0.0,
    decay_frac_power: float = 3.0,
    grad_coeff: Optional[float] = None,
) -> Phase:
    n_warmup_steps = get_warmup_steps(phase_steps, warmup_frac, n_warmup_steps)

    final_lr = final_lr_frac * max_lr
    return Phase(
        n_steps=phase_steps,
        preclip_lr=preclip_lr,
        lr_sched=optax.warmup_cosine_decay_schedule(1e-9 * max_lr, max_lr, n_warmup_steps, phase_steps, final_lr),
        wl_sched=wl.truncated_transition(
            prev_trunc_steps, new_trunc_steps, phase_steps, max_power, T, decay_frac_power=decay_frac_power
        ),
        grad_coeff=grad_coeff,
    )


class PhaseScheduler:
    def __init__(self, phases: List[Phase]):
        self._phases = phases

        # Compute the phase boundaries.
        self._phase_steps = np.array([phase.n_steps for phase in phases])
        self._boundaries = np.cumsum(self._phase_steps)[:-1]
        self._offsets = np.concatenate([[0], self._boundaries])

    def get_phase_idx(self, idx: int) -> int:
        phase_idx = np.searchsorted(self._boundaries, idx, side="right")

        return phase_idx

    def get_phase(self, idx: int) -> Tuple[int, Phase]:
        phase_idx = self.get_phase_idx(idx)
        phase = self._phases[phase_idx]
        offset = self._offsets[phase_idx]

        return offset, phase

    def get_preclip_lr(self, idx: int) -> float:
        offset, phase = self.get_phase(idx)
        return phase.preclip_lr

    def get_lr(self, idx: int) -> float:
        offset, phase = self.get_phase(idx)
        return phase.lr_sched(idx - offset)

    def get_wl_sched(self, idx: int) -> wl.LossWeightSchedule:
        offset, phase = self.get_phase(idx)
        return wl.offset_sched(phase.wl_sched, offset)

    def get_weights(self, idx: int) -> ja:
        offset, phase = self.get_phase(idx)
        return phase.wl_sched(idx - offset)

    def get_train_steps(self) -> int:
        return int(np.sum(self._phase_steps))

    def get_grad_coeff_scale(self, idx: int) -> Optional[float]:
        offset, phase = self.get_phase(idx)

        if (idx - offset) == 0:
            return phase.grad_coeff

        return None
