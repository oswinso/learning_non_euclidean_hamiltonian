from typing import List, TypeVar

import jax
import jax.numpy as jnp

from utils.jax_utils import pytree_stack

_T = TypeVar("_T")
ja = jnp.ndarray


def combine_weighted(grads: List[_T], weights: ja) -> _T:
    assert weights.ndim == 1
    assert len(grads) == weights.shape[0]

    grads_tree = pytree_stack(grads, axis=0)
    # (n_grads, *) -> (*, )
    weighted = jax.tree_map(lambda x: jnp.sum(jnp.einsum("i...,i->i...", x, weights), axis=0), grads_tree)

    return weighted
