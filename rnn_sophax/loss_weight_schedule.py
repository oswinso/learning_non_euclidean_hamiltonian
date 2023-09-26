import jax.numpy as jnp
import optax
from attr import s


@s(auto_attribs=True, slots=True, auto_exc=True, auto_detect=True, order=False)
class LossWeightCfg:
    base: float
    max_power: float
    steps: int


def exponential_schedule(init_value: float, base: float, max_power: float, steps: int) -> optax.Schedule:
    assert steps > 0

    def schedule(count: int):
        count = jnp.clip(count, 0, steps)
        frac = count / steps
        power = frac * max_power
        return init_value * (base ** power)

    return schedule


def get_loss_weight_schedule(cfg: LossWeightCfg) -> optax.Schedule:
    schedule = exponential_schedule(1, cfg.base, cfg.max_power, cfg.steps)

    return schedule
