from typing import Optional, Protocol, Sequence, Tuple

import chex
from jax import numpy as jnp

from utils.types import ja


class LossWeightSchedule(Protocol):
    def __call__(self, count: int) -> ja:
        ...


def truncated_loss(n_steps: int, T: int) -> LossWeightSchedule:
    def schedule(count: int) -> ja:
        weights = jnp.zeros(T)
        weights = weights.at[:n_steps].set(1.0)
        return weights

    return schedule


def one_step(T: int) -> LossWeightSchedule:
    return truncated_loss(1, T)


def truncated_transition(
    prev_n_steps: int,
    new_n_steps: int,
    transition: int,
    max_power: float,
    T: int,
    decay_frac_power: float = 1.0,
    coeff_frac_power: float = 1.0,
) -> LossWeightSchedule:
    def schedule(count: int) -> ja:
        weights = jnp.zeros(T)
        weights = weights.at[:prev_n_steps].set(1.0)

        # Exponential decay
        count = jnp.clip(count + 1e-8, 1e-8, transition)
        frac = count / float(transition)
        decay = (frac ** decay_frac_power) * max_power
        decay = jnp.clip(decay, 1e-8)
        exp_coeff = frac ** coeff_frac_power
        exp_decay = jnp.exp(-jnp.arange(T - prev_n_steps) / decay)
        weights = weights.at[prev_n_steps:].set(exp_coeff * exp_decay)

        # Zero out the tail.
        weights = weights.at[new_n_steps:].set(0)

        return weights

    return schedule


def exponential_weight(base: float, max_power: float, steps: float, T: int) -> LossWeightSchedule:
    def schedule(count: int) -> ja:
        count = jnp.clip(count, 0, steps)
        frac = count / steps
        power = frac * max_power
        decay = base ** power

        weights = jnp.exp(-jnp.arange(T) / decay)
        return weights

    return schedule


def constant_weight(T: int) -> LossWeightSchedule:
    def schedule(count: int) -> ja:
        return jnp.ones(T)

    return schedule


def multiply_schedules(schedules: Sequence[LossWeightSchedule]) -> LossWeightSchedule:
    def _schedule(step: int) -> ja:
        outputs = [schedule(step) for schedule in schedules]
        stacked = jnp.stack(outputs, axis=0)

        return jnp.prod(stacked, axis=0)

    return _schedule


def join_schedules(schedules: Sequence[LossWeightSchedule], boundaries: Sequence[int]) -> LossWeightSchedule:
    assert len(boundaries) + 1 >= len(schedules)

    def _schedule(step: int) -> ja:
        output = schedules[0](step)
        for boundary, schedule in zip(boundaries, schedules[1:]):
            output = jnp.where(step < boundary, output, schedule(step - boundary))

        return output

    return _schedule


def offset_sched(schedule: LossWeightSchedule, offset: int) -> LossWeightSchedule:
    def _schedule(step: int) -> ja:
        return schedule(step - offset)

    return _schedule


def get_loss_weights(idx: int, schedule: LossWeightSchedule) -> Tuple[ja, Optional[int]]:
    weights = schedule(idx)

    # Append 0 to the beginning since the initial conditions will always match.
    weights = jnp.concatenate([jnp.zeros(1), weights], axis=0)

    # Normalize so that it sums to 1.
    weights = weights / weights.sum()

    # Find the first zero. We can then integrate only until the last nonzero.
    first_zero = jnp.nonzero(weights[1:] < 1e-10)[0]
    if first_zero.size > 0:
        first_zero = first_zero[0] + 1
    else:
        first_zero = None

    return weights, first_zero
