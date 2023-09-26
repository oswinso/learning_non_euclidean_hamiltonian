import logging
from typing import Optional

import haiku as hk
import ipdb
import jax.numpy as jnp
import numpy as np
import optax

from utils.types import ja

log = logging.getLogger(__file__)


class InputNormalizer(hk.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def __call__(self, x: ja) -> ja:
        mean = hk.get_parameter("mean", x.shape, x.dtype, init=jnp.zeros)
        std = hk.get_parameter("std", x.shape, x.dtype, init=jnp.ones)

        # return x
        tmp = x - mean
        # tmp = tmp / std
        return tmp

        # stds = (
        #     [1.0] * 5
        #     + [
        #         7e-1,
        #         7e-1,
        #         2e-9,
        #         7e-1,
        #         7e-1,
        #         2e-11,
        #         3e-9,
        #         3e-9,
        #         1e-0,
        #         3e-8,
        #         7e-1,
        #         7e-1,
        #         1e-8,
        #         7e-1,
        #         7e-1,
        #         1e-10,
        #         7e-3,
        #         7e-3,
        #     ]
        #     + [1e-10] * 18
        # )

        # stds = [1.0] * 5 + [1.0] * 18 + [1e-1] * 18
        # stds = jnp.array(stds)
        #
        # return tmp / stds

        # weights = 1 / stds
        # weights = np.ones(41)
        #
        # tmp = tmp * weights
        # return tmp


def set_normalization(batched_data: ja, params: optax.Params) -> optax.Params:
    # (batch, nx)
    assert batched_data.ndim == 2

    normalizer_key = [key for key in params.keys() if key.rsplit("/", 1)[1] == "normalizer"]

    if len(normalizer_key) == 0:
        log.warning("No normalization params found!")
        return params

    key = normalizer_key[0]

    mean = jnp.mean(batched_data, axis=0)
    std = jnp.std(batched_data, axis=0)

    print("set_normalization min_std={}".format(std.min()))

    # if std == 0, set it to 1.
    std = jnp.where(std == 0.0, 1.0, std)

    # Clip std to make sure it's not too tiny.
    min_std = 1e-10
    clipped_std = jnp.clip(std, a_min=min_std)

    assert params[key]["mean"].shape == mean.shape
    assert params[key]["std"].shape == clipped_std.shape

    params[key]["mean"] = jnp.array(params[key]["mean"]).at[:].set(mean)
    params[key]["std"] = clipped_std

    return params
