from typing import List

import haiku as hk
import ipdb
import jax.nn
import jax.numpy as jnp

from utils.haiku_utils import ActivatedLinear, ParamNames
from utils.types import ja


def test_activated_linear():
    param_names: List[ParamNames] = []

    def fn(x: ja) -> ja:
        lin = ActivatedLinear(3, jax.nn.tanh)
        out = lin(x)

        nonlocal param_names
        param_names.append(lin.get_param_names())

        return out

    nn = hk.without_apply_rng(hk.transform(fn))

    rng = hk.PRNGSequence(1241)
    x = jnp.ones(5)
    params = nn.init(next(rng), x)

    # There should be 3 param names now.
    assert len(param_names[0].full_names) == 3

    orig_result = nn.apply(params, x)

    # Scale gradients.
    new_params = ActivatedLinear.adjust_scale(params, param_names[0], 10.0)

    new_result = nn.apply(new_params, x)

    assert jnp.allclose(orig_result, new_result)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        test_activated_linear()
