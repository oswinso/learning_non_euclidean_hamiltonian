import jax.numpy as jnp
import jax.scipy
from jax import custom_jvp

from utils.types import ja


@custom_jvp
def int_tanh(x: ja) -> ja:
    return jnp.log(jnp.cosh(x))


@int_tanh.defjvp
def int_tanh_jvp(primals, tangents):
    (x,) = primals
    (t,) = tangents
    return int_tanh(x), jnp.tanh(x) * t


@custom_jvp
def int_gelu(x: ja) -> ja:
    erf_term = jax.scipy.special.erf(x / jnp.sqrt(2))
    x_sq = jnp.square(x)
    return (x * (jnp.exp(-x_sq / 2) * jnp.sqrt(2 / jnp.pi) + x) + (x_sq - 1) * erf_term) / 4.0


@int_gelu.defjvp
def int_gelu_jvp(primals, tangents):
    (x,) = primals
    (t,) = tangents

    erf_term = jax.scipy.special.erf(x / jnp.sqrt(2))
    x_sq = jnp.square(x)

    int_gelu_term = (x * (jnp.exp(-x_sq / 2) * jnp.sqrt(2 / jnp.pi) + x) + (x_sq - 1) * erf_term) / 4.0
    deriv_term = x / 2 * (1 + erf_term) * t

    return int_gelu_term, deriv_term


def identity_act(x: ja) -> ja:
    return x
