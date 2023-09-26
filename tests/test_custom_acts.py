import jax
import jax.numpy as jnp

from utils.custom_acts import int_tanh
from utils.types import ja


def test_int_tanh():
    xs = jnp.linspace(-10, 10, num=50)

    def f(x: ja) -> ja:
        return int_tanh(x).sum()

    grad_int_tanh = jax.grad(f)
    evals = grad_int_tanh(xs)

    assert jnp.allclose(evals, jnp.tanh(xs))
