from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import linear_util as lu
from jax._src.api import (
    _check_callable,
    _check_input_dtype_jacrev,
    _check_output_dtype_jacrev,
    _jacrev_unravel,
    _std_basis,
    _vjp,
)
from jax._src.api_util import argnums_partial
from jax._src.ops.scatter import Index
from jax.config import config
from jax.tree_util import tree_map, tree_structure, tree_transpose

from utils.dataclass_utils import fields_dict, is_dc_or_attr
from utils.types import ja

_T = TypeVar("_T")


def jax_use_double(val: bool = True) -> None:
    config.update("jax_enable_x64", val)


def jax_use_cpu() -> None:
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_platforms", "cpu")


def get_rng_key(key: Optional[chex.PRNGKey], seed: int) -> chex.PRNGKey:
    if key is not None:
        return key

    return jax.random.PRNGKey(seed)


def get_rng_stream(key: Optional[chex.PRNGKey], seed: int) -> hk.PRNGSequence:
    return hk.PRNGSequence(get_rng_key(key, seed))


def concat_at_end(arr1: ja, arr2: ja, axis: int) -> ja:
    # The shapes of arr1 and arr2 should be the same without the dim at axis for arr1.
    arr1_shape = list(arr1.shape)
    del arr1_shape[axis]
    assert np.all(np.array(arr1_shape) == np.array(arr2.shape))

    return jnp.concatenate([arr1, jnp.expand_dims(arr2, axis=axis)], axis=axis)


def add_batch(nest: _T, batch_size: Optional[int]) -> _T:
    def broadcast(x: jnp.ndarray):
        return jnp.broadcast_to(x, (batch_size,) + x.shape)

    return jax.tree_map(broadcast, nest)


def flat01(arr: ja) -> ja:
    shape = arr.shape
    new_shape = (shape[0] * shape[1], *shape[2:])
    return arr.reshape(new_shape)


def block(xs: List[List[ja]]) -> ja:
    assert isinstance(xs, list)
    assert len(xs) != 0

    n_rows, n_cols = len(xs), len(xs[0])

    rows = [jnp.stack(row, axis=-1) for row in xs]
    mat = jnp.stack(rows, axis=-2)
    assert mat.shape[-2:] == (n_rows, n_cols)

    return mat


def scalarize(fn):
    def _fn(*args, **kwargs) -> ja:
        orig_out: ja = fn(*args, **kwargs)
        assert orig_out.shape == (1,)

        return orig_out.squeeze()

    return _fn


TreeType = TypeVar("TreeType")
Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


def serial_scan(
    f: Callable[[Carry, X], Tuple[Carry, Y]],
    init: Carry,
    xs: X,
    reverse: bool = False,
    length: Optional[int] = None,
    unroll: int = 1,
) -> Tuple[Carry, Y]:
    carry = init
    ys = []

    if length is None:
        args_flat, in_tree = jax.tree_util.tree_flatten(xs)
        axis_size = args_flat[0].shape[0]

        if reverse:
            xs = pytree_reverse(xs)

        for t in range(axis_size):
            x = pytree_index(xs, t)
            carry, y = f(carry, x)
            ys.append(y)
    else:
        for t in range(length):
            carry, y = f(carry, None)
            ys.append(y)

    ys = pytree_stack(ys)

    if reverse:
        ys = pytree_reverse(ys)

    return carry, ys


def pytree_index(tree: _T, idx: Index) -> _T:
    def f(arr: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(arr, idx, axis=0)

    return jax.tree_map(f, tree)


def pytree_stack(trees: List[_T], axis: int = 0) -> _T:
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l, axis=axis) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def pytree_reduce(trees: List[TreeType], op=jnp.stack, **op_kwargs) -> TreeType:
    """
    :param trees:
    :param op:
    :param op_kwargs:
    :return:
    """
    leaves_list = []
    treedef_list = []

    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [op(leaves, **op_kwargs) for leaves in grouped_leaves]

    return treedef_list[0].unflatten(result_leaves)


def pytree_reverse(tree: TreeType) -> TreeType:
    def f(arr: jnp.ndarray) -> jnp.ndarray:
        return jnp.flip(arr, axis=0)

    return jax.tree_map(f, tree)


def register_dataclass(clz: Union[_T, List[_T]]):
    if isinstance(clz, list):
        for single_clz in clz:
            _register_dataclass(single_clz)
    else:
        _register_dataclass(clz)


def _register_dataclass(clz: _T):
    assert is_dc_or_attr(clz)

    meta_fields = []
    data_fields = []
    for name, field_info in fields_dict(clz).items():
        is_pytree_node = field_info.metadata.get("pytree_node", True)
        if is_pytree_node:
            data_fields.append(name)
        else:
            meta_fields.append(name)

    def iterate_clz(x):
        meta = tuple(getattr(x, _name) for _name in meta_fields)
        data = tuple(getattr(x, _name) for _name in data_fields)
        return data, meta

    def clz_from_iterable(meta, data):
        meta_args = tuple(zip(meta_fields, meta))
        data_args = tuple(zip(data_fields, data))
        kwargs = dict(meta_args + data_args)
        return clz(**kwargs)

    jax.tree_util.register_pytree_node(clz, iterate_clz, clz_from_iterable)


def global_norm(updates: _T) -> _T:
    """Compute the global norm across a nested structure of tensors."""
    return jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(updates)]))


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def value_and_jacrev(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable:
    _check_callable(fun)

    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args, require_static_args_hashable=False)
        tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
        if not has_aux:
            y, pullback = _vjp(f_partial, *dyn_args)
        else:
            y, pullback, aux = _vjp(f_partial, *dyn_args, has_aux=True)
        tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
        jac = jax.vmap(pullback)(_std_basis(y))
        jac = jac[0] if isinstance(argnums, int) else jac
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = tree_map(partial(_jacrev_unravel, y), example_args, jac)
        jac_tree = tree_transpose(tree_structure(example_args), tree_structure(y), jac_tree)

        if not has_aux:
            return y, jac_tree
        else:
            return (y, aux), jac_tree

    return jacfun
