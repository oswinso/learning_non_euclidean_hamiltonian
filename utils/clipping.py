import logging
from typing import List, NamedTuple, Optional, Tuple, Union

import chex
import ipdb
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from optax._src import base, linear_algebra

from utils.types import ja

log = logging.getLogger(__file__)


class ClipByGlobalNormState(NamedTuple):
    g_norm_before: ja
    g_norm_after: ja


AdaptiveGradClipState = base.EmptyState


def unitwise_norm(x: ja) -> chex.Array:
    """Computes norms of each output unit separately."""
    if jnp.squeeze(x).ndim <= 1:  # Scalars and vectors
        squared_norm = jnp.sum(jnp.square(x), keepdims=True)
    # Note that this assumes parameters with a shape of length 3 are multihead
    # linear parameters--if you wish to apply AGC to 1D convs, you may need
    # to modify this line.
    elif x.ndim in (2, 3):  # Linear layers of shape IO or multihead linear
        squared_norm = jnp.sum(jnp.square(x), axis=0, keepdims=True)
    elif x.ndim == 4:  # Conv kernels of shape HWIO
        squared_norm = jnp.sum(jnp.square(x), axis=(0, 1, 2), keepdims=True)
    else:
        raise ValueError(f"Expected parameter with shape in {1, 2, 3, 4}, got {x.shape}.")
    chex.assert_is_broadcastable(squared_norm.shape, x.shape)
    return jnp.broadcast_to(jnp.sqrt(squared_norm), x.shape)


def unitwise_clip(g_norm: ja, max_norm: ja, grad: ja, div_eps: float = 1e-6) -> ja:
    """Applies gradient clipping unit-wise."""
    # This little max(., div_eps) is distinct from the normal eps and just
    # prevents division by zero. It technically should be impossible to engage.
    clipped_grad = grad * (max_norm / jnp.maximum(g_norm, div_eps))
    chex.assert_equal_shape((g_norm, max_norm, grad, clipped_grad))
    return jnp.where(g_norm < max_norm, grad, clipped_grad)


def adaptive_grad_clip(clipping: float = 1e-2, eps: float = 1e-3) -> base.GradientTransformation:
    log.info("adaptive_grad_clip clipping: {}, eps: {}".format(clipping, eps))

    def init_fn(_):
        del _
        return AdaptiveGradClipState()

    def update_fn(
        updates: optax.Updates, state: AdaptiveGradClipState, params: optax.Params = None
    ) -> Tuple[optax.Params, AdaptiveGradClipState]:
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)

        g_norm, p_norm = jax.tree_map(unitwise_norm, (updates, params))
        # Maximum allowable norm.
        max_norm = jax.tree_map(lambda x: clipping * jnp.maximum(x, eps), p_norm)
        # If grad norm > clipping * param_norm, rescale.
        updates = jax.tree_map(unitwise_clip, g_norm, max_norm, updates)

        return updates, state

    return base.GradientTransformation(init_fn, update_fn)


def clip_by_global_norm(max_norm: float, max_delta: float) -> base.GradientTransformation:
    log.info("clip_by_global_norm max_norm: {}, max_delta: {}".format(max_norm, max_delta))

    def init_fn(_):
        return ClipByGlobalNormState(jnp.zeros(1), jnp.zeros(1))

    def update_fn(
        updates: optax.Updates, state: ClipByGlobalNormState, params: optax.Params = None
    ) -> Tuple[optax.Params, ClipByGlobalNormState]:
        del params
        g_norm_before = linear_algebra.global_norm(updates)

        # Also clip individual parameters.
        flat_updates, _ = ravel_pytree(updates)
        # q_delta = jnp.quantile(flat_updates, 0.99)
        # q_delta = jnp.maximum(q_delta, max_delta)
        updates = jax.tree_map(lambda g: jnp.clip(g, a_min=-max_delta, a_max=max_delta), updates)

        g_norm_after = linear_algebra.global_norm(updates)

        # TODO(b/163995078): revert back to the following (faster) implementation
        # once analysed how it affects backprop through update (e.g. meta-gradients)
        # g_norm = jnp.maximum(max_norm, g_norm)
        # updates = jax.tree_map(lambda t: (t / g_norm) * max_norm, updates)
        trigger = jnp.squeeze(g_norm_after < max_norm)
        chex.assert_shape(trigger, ())  # A scalar.

        updates = jax.tree_map(lambda t: jax.lax.select(trigger, t, (t / g_norm_after) * max_norm), updates)

        state = ClipByGlobalNormState(g_norm_before, g_norm_after)
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)


def get_global_norm(
    opt_state: Union[optax.InjectHyperparamsState, optax.OptState]
) -> Tuple[Optional[ja], Optional[ja]]:
    global_norms = _get_global_norm(opt_state)

    if len(global_norms) == 1:
        return global_norms[0]

    return None, None


def _get_global_norm(opt_state: Union[optax.InjectHyperparamsState, optax.OptState]) -> List[Tuple[ja, ja]]:
    if isinstance(opt_state, optax.InjectHyperparamsState):
        return _get_global_norm(opt_state.inner_state)

    if isinstance(opt_state, ClipByGlobalNormState):
        return [(opt_state.g_norm_before, opt_state.g_norm_after)]

    if isinstance(opt_state, list) or isinstance(opt_state, tuple):
        return [(s.g_norm_before, s.g_norm_after) for s in opt_state if isinstance(s, ClipByGlobalNormState)]
        # tmps = []
        # for elem in opt_state:
        #     tmps.extend(_get_global_norm(elem))
        # return tmps
    else:
        return []
