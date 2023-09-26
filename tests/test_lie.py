import jax.numpy as jnp
import numpy as np

from utils import lie
from utils.types import ja


def test_hat():
    tmp = jnp.array([1, 2, 3])
    hat = lie.hat(tmp)

    assert jnp.all(
        hat
        == np.array(
            [
                [0, -3, 2],
                [3, 0, -1],
                [-2, 1, 0],
            ]
        )
    )

def test_vee():
    tmp = jnp.array([1, 2, 3])

    vee_hat = lie.vee(lie.hat(tmp))

    assert jnp.all(vee_hat == tmp)
