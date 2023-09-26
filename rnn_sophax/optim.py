from typing import Tuple

import haiku as hk
import ipdb
import jax
import optax
from attr import s

from utils.clipping import adaptive_grad_clip, clip_by_global_norm


@s(auto_attribs=True, slots=True, auto_exc=True, auto_detect=True, order=False)
class OptimCfg:
    grad_clip: float = 1.0


def not_bias(module_name, name, value) -> bool:
    return not (module_name == "group_norm" or name == "b")


def not_bias_mask(p):
    return hk.data_structures.map(not_bias, p)


def get_optimizer(cfg: OptimCfg) -> optax.GradientTransformation:
    @optax.inject_hyperparams
    def optimizer(lr: float, preclip_lr: float) -> optax.GradientTransformation:
        # coeff = 50
        coeff = 10

        clip = clip_by_global_norm(cfg.grad_clip, coeff * cfg.grad_clip)
        # clip = adaptive_grad_clip(clipping=1e-2, eps=1e-3)
        # optax.adaptive_grad_clip()

        radam_cfg = dict(b1=0.9, b2=0.999, eps=1e-8, threshold=5.0)
        adam_cfg = dict(b1=0.9, b2=0.999, eps=1e-8)
        belief_cfg = dict(b1=0.9, b2=0.999, eps=1e-16, eps_root=1e-16)
        weight_decay = 1e-2

        # I'm guessing this is so we get a descent direction?
        flip_sign = True
        m = -1 if flip_sign else 1

        optim = optax.chain(
            optax.scale(preclip_lr),
            clip,
            # optax.scale_by_belief(**belief_cfg),
            # optax.scale_by_radam(**radam_cfg),
            optax.scale_by_adam(**adam_cfg),
            # Don't weight decay bias.
            optax.masked(optax.add_decayed_weights(weight_decay), not_bias_mask),
            optax.scale(m * lr),
        )

        # optim = optax.lookahead(optim, sync_period=6, slow_step_size=0.5)
        #
        return optim

    return optimizer(1.0, 1.0)
