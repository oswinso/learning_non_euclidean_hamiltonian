import jax.numpy as jnp
import numpy as np

from utils.jax_utils import block
from utils.types import ja


def scalar(x: float) -> ja:
    return jnp.full(1, x)


def test_block():
    tmp = [[scalar(1), scalar(2), scalar(3)], [scalar(4), scalar(5), scalar(6)]]
    res = np.array(block(tmp))

    assert jnp.all(res == np.array([[1, 2, 3], [4, 5, 6]]))
