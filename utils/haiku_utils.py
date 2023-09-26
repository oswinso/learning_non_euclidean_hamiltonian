import logging
import math
from enum import Enum
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar, Union

import haiku as hk
import ipdb
import jax
import jax.numpy as jnp
import numpy as np
import optax
from attr import s
from haiku._src.base import current_bundle_name
from jax import lax

from utils.custom_acts import identity_act, int_tanh
from utils.jax_utils import register_dataclass
from utils.types import ja

log = logging.getLogger(__file__)

ActFn = Callable[[ja], ja]


class Activation(Enum):
    Relu = 0
    Tanh = 1
    Swish = 2
    HardSwish = 3
    Gelu = 4
    IntTanh = 5

    def get(self) -> ActFn:
        return get_act(self)


@s(slots=True, auto_attribs=True, auto_detect=True)
class MLPCfg:
    hidden: List[int]
    act: Union[Activation, List[Activation]]
    w_init_gain: float = 1.0
    zero_final_weights: bool = False
    final_bias: bool = False
    final_weight_coeff: Optional[float] = None

    def get_acts(self) -> Union[ActFn, List[ActFn]]:
        acts = self.act
        if isinstance(acts, Activation):
            return acts.get()

        return [act.get() for act in acts]


def get_act(act: Activation) -> Callable[[ja], ja]:
    import jax

    if act == Activation.Relu:
        return jax.nn.relu
    elif act == Activation.Tanh:
        return jax.nn.tanh
    elif act == Activation.Swish:
        return jax.nn.swish
    elif act == Activation.HardSwish:
        return jax.nn.hard_swish
    elif act == Activation.Gelu:
        return jax.nn.gelu
    elif act == Activation.IntTanh:
        return int_tanh
    else:
        raise ValueError(f"Unknown act {act}")


def get_output_sizes(output_dim: Optional[int], params: MLPCfg) -> List[int]:
    if output_dim is not None:
        assert isinstance(output_dim, int)
        output_dim = [output_dim]
    else:
        output_dim = []

    output_sizes = params.hidden + output_dim

    return output_sizes


class ResBlock(hk.Module):
    def __init__(self, hidden_dim: int, act: ActFn):
        super().__init__()

        layers = []
        widths = [hidden_dim] * 3
        for width in widths:
            layers.append(hk.Linear(width))
            layers.append(act)
        self.block = hk.Sequential(layers)

    def __call__(self, inputs: ja) -> ja:
        return self.block(inputs)


class ResnetFC(hk.Module):
    def __init__(self, hidden_dim: int, n_res_blocks: int, act: ActFn, name: Optional[str] = None):
        super().__init__(name=name)

        self.enc = hk.Linear(hidden_dim)
        self.res_blocks = [ResBlock(hidden_dim, act) for _ in range(n_res_blocks)]
        self.norms = [hk.GroupNorm(32, axis=-1) for _ in range(n_res_blocks)]

    def __call__(self, inputs: ja) -> ja:
        h = self.enc(inputs)

        for res_block, norm in zip(self.res_blocks, self.norms):
            h = (h + res_block(norm(h))) / jnp.sqrt(2)

        return h


class MLP(hk.Module):
    def __init__(
        self,
        output_sizes: Iterable[int],
        w_init_gain: Optional[float] = 1.0,
        with_bias: bool = True,
        activation: Union[ActFn, List[ActFn]] = jax.nn.relu,
        activate_final: bool = False,
        input_size: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        layers = []
        output_sizes = tuple(output_sizes)

        # If it's not a list, then use the same activation for all layers.
        if not isinstance(activation, list):
            n_activations = len(output_sizes) + 1 if activate_final else len(output_sizes)
            activation = [activation] * n_activations

        self.activation = activation
        self.activate_final = activate_final

        for index, output_size in enumerate(output_sizes):
            if index > 1:
                layer_input_size = output_sizes[index - 1]
            else:
                layer_input_size = input_size if input_size is not None else 10
            stddev = w_init_gain / math.sqrt(layer_input_size)

            # w_init = hk.initializers.TruncatedNormal(stddev=stddev)
            w_init = hk.initializers.Orthogonal(scale=stddev)

            b_init = jnp.zeros
            layers.append(
                hk.Linear(
                    output_size=output_size, w_init=w_init, b_init=b_init, with_bias=with_bias, name="linear_%d" % index
                )
            )
        self.layers = tuple(layers)
        self.output_size = output_sizes[-1] if output_sizes else None

    def __call__(
        self,
        inputs: jnp.ndarray,
        dropout_rate: Optional[float] = None,
        rng=None,
    ) -> jnp.ndarray:
        """Connects the module to some inputs.

        Args:
          inputs: A Tensor of shape ``[batch_size, input_size]``.
          dropout_rate: Optional dropout rate.
          rng: Optional RNG key. Require when using dropout.

        Returns:
          The output of the model of size ``[batch_size, output_size]``.
        """
        if dropout_rate is not None and rng is None:
            raise ValueError("When using dropout an rng key must be passed.")
        elif dropout_rate is None and rng is not None:
            raise ValueError("RNG should only be passed when using dropout.")

        rng = hk.PRNGSequence(rng) if rng is not None else None
        num_layers = len(self.layers)

        out = inputs
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < (num_layers - 1) or self.activate_final:
                # Only perform dropout if we are activating the output.
                if dropout_rate is not None:
                    out = hk.dropout(next(rng), dropout_rate, out)
                out = self.activation[i](out)

        return out


@s(slots=True, auto_attribs=True, auto_detect=True)
class ParamNames:
    bundle_name: str
    scale_name: str
    param_names: List[str]


class ActivatedMLP(hk.Module):
    def __init__(
        self,
        output_sizes: Iterable[int],
        w_init_gain: Optional[float] = 1.0,
        with_bias: bool = True,
        with_bias_final: bool = True,
        activation: Union[ActFn, List[ActFn]] = jax.nn.relu,
        activate_final: bool = False,
        input_size: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        layers = []
        output_sizes = tuple(output_sizes)

        # If it's not a list, then use the same activation for all layers.
        if not isinstance(activation, list):
            activations = [activation] * len(output_sizes)

            if activate_final:
                activations.append(activation)

        else:
            assert isinstance(activation, list)
            activations = activation

        if not activate_final and len(activations) == len(output_sizes) - 1:
            activations.append(identity_act)

        self.activation = activations
        self.activate_final = activate_final

        for index, output_size in enumerate(output_sizes):
            if index > 1:
                layer_input_size = output_sizes[index - 1]
            else:
                layer_input_size = input_size if input_size is not None else 10
            stddev = w_init_gain / math.sqrt(layer_input_size)

            # w_init = hk.initializers.TruncatedNormal(stddev=stddev)
            w_init = hk.initializers.Orthogonal(scale=stddev)

            b_init = jnp.zeros

            is_final = index == len(output_sizes) - 1
            layers.append(
                ActivatedLinear(
                    output_size=output_size,
                    activation=self.activation[index],
                    w_init=w_init,
                    b_init=b_init,
                    with_bias=with_bias_final if is_final else with_bias,
                    name="linear_%d" % index,
                )
            )

        self.layers = tuple(layers)
        self.output_size = output_sizes[-1] if output_sizes else None

    def get_param_names(self) -> List[ParamNames]:
        return [layer.get_param_names() for layer in self.layers]

    def __call__(
        self,
        inputs: jnp.ndarray,
        dropout_rate: Optional[float] = None,
        rng=None,
    ) -> jnp.ndarray:
        """Connects the module to some inputs.

        Args:
          inputs: A Tensor of shape ``[batch_size, input_size]``.
          dropout_rate: Optional dropout rate.
          rng: Optional RNG key. Require when using dropout.

        Returns:
          The output of the model of size ``[batch_size, output_size]``.
        """
        if dropout_rate is not None and rng is None:
            raise ValueError("When using dropout an rng key must be passed.")
        elif dropout_rate is None and rng is not None:
            raise ValueError("RNG should only be passed when using dropout.")

        rng = hk.PRNGSequence(rng) if rng is not None else None
        num_layers = len(self.layers)

        out = inputs
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < (num_layers - 1) or self.activate_final:
                # Only perform dropout if we are activating the output.
                if dropout_rate is not None:
                    out = hk.dropout(next(rng), dropout_rate, out)

        return out


class ActivatedLinear(hk.Module):
    def __init__(
        self,
        output_size: int,
        activation: ActFn,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.input_size = None
        self.activation = activation
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros

    def get_param_names(self) -> ParamNames:
        log.info("current_bundle_name: {}".format(current_bundle_name()))
        bundle_name = current_bundle_name()

        param_names = ["w", "b"] if self.with_bias else ["w"]
        return ParamNames(bundle_name, "scale", param_names)

    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        precision: Optional[lax.Precision] = None,
    ) -> jnp.ndarray:
        """Computes a linear transform of the input."""
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)

        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

        out = jnp.dot(inputs, w, precision=precision)

        if self.with_bias:
            b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        # Rescale the output before
        output_scale = hk.get_parameter("scale", [1], dtype, init=jnp.ones)
        out = self.activation(output_scale * out)

        return out

    @staticmethod
    def adjust_scale(
        params: optax.Params, frozen_params: optax.Params, param_names: ParamNames, coeff: float
    ) -> Tuple[optax.Params, optax.Params]:
        """Adjust the scale. Coeff smaller than 1 will make gradients smaller and
        make the weights larger correspondingly."""
        new_params = params.copy()
        new_frozen_params = frozen_params.copy()

        bundle_name = param_names.bundle_name
        scale_name = param_names.scale_name

        new_frozen_params[bundle_name][param_names.scale_name] = new_frozen_params[bundle_name][scale_name] * coeff

        for param_name in param_names.param_names:
            new_params[bundle_name][param_name] = new_params[bundle_name][param_name] / coeff

        return new_params, new_frozen_params


def scale_init(init: hk.initializers.Initializer, scale: float) -> None:
    if isinstance(init, hk.initializers.RandomNormal):
        init.stddev = init.stddev * scale
    elif isinstance(init, hk.initializers.TruncatedNormal):
        init.stddev = init.stddev * scale
    elif isinstance(init, hk.initializers.Orthogonal):
        init.scale = init.scale * scale
    else:
        raise RuntimeError("Unknown init fn {}!".format(init))


def get_mlp(output_dim: Optional[int], params: MLPCfg, std_init_input_size: int, name: str) -> MLP:
    output_sizes = get_output_sizes(output_dim, params)

    net = MLP(
        output_sizes,
        w_init_gain=params.w_init_gain,
        activation=params.get_acts(),
        input_size=std_init_input_size,
        name=name,
    )

    if params.zero_final_weights:
        net.layers[-1].w_init = jnp.zeros
    elif params.final_weight_coeff is not None:
        scale_init(net.layers[-1].w_init, params.final_weight_coeff)

    return net


def get_activated_mlp(output_dim: Optional[int], params: MLPCfg, input_size: int, name: str) -> ActivatedMLP:
    output_sizes = get_output_sizes(output_dim, params)

    net = ActivatedMLP(
        output_sizes, w_init_gain=params.w_init_gain, activation=params.get_acts(), input_size=input_size, name=name
    )

    if params.zero_final_weights:
        net.layers[-1].w_init = jnp.zeros

    return net


_Transformed = TypeVar("_Transformed", hk.Transformed, hk.MultiTransformed)


def without_apply_rng(f: _Transformed) -> _Transformed:
    """Converts ``MultiTransformedWithState`` to ``MultiTransformed``."""

    if isinstance(f, hk.Transformed):
        return hk.without_apply_rng(f)

    assert isinstance(f, hk.MultiTransformed)

    def apply_without_rng(orig_apply_fn) -> Any:
        def apply_fn(params: hk.Params, *args, **kwargs):
            out = orig_apply_fn(params, None, *args, **kwargs)
            return out

        return apply_fn

    apply_fns = jax.tree_map(apply_without_rng, f.apply)

    return hk.MultiTransformed(f.init, apply_fns)


register_dataclass(ParamNames)
