import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal, Optional

from flaxsr.losses.utils import reduce_fn, Reduce, Loss, load_vgg19_params
from flaxsr._utils import register


Pytree = any


def _conv(x: jnp.ndarray, weights: jnp.ndarray, bias: jnp.ndarray):
    x = lax.conv_general_dilated(
        x, weights, window_strides=(1, 1), padding='SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    x = x + bias
    return x


@partial(jax.jit, static_argnums=(2,))
def _get_feats_from_vgg19(x: jnp.ndarray, params: Pytree, before_act: bool) -> Sequence[jnp.ndarray]:
    max_pool_indices = [
        1, 3, 7, 11, 15
    ]

    outputs = []
    for i, (w, b) in enumerate(params):
        x = _conv(x, w, b)
        if before_act:
            outputs.append(x)
            x = nn.relu(x)
        else:
            x = nn.relu(x)
            outputs.append(x)
        if i in max_pool_indices:
            x = nn.max_pool(x, (2, 2), (2, 2), 'SAME')
        else:
            x = x  # for jax.jit
    outputs.append(x)
    return outputs


@partial(jax.jit, static_argnums=(3, 4, 5,))
def vgg_loss(
        sr: jnp.ndarray, hr: jnp.ndarray, vgg_params: Pytree, feats_from: Sequence[int],
        before_act: bool = False, reduce: str | Reduce = 'mean'
) -> jnp.ndarray:
    hr_feats = _get_feats_from_vgg19(hr, vgg_params, before_act)
    sr_feats = _get_feats_from_vgg19(sr, vgg_params, before_act)

    loss = jnp.zeros(())
    for i, (hr_feat, sr_feat) in enumerate(zip(hr_feats, sr_feats)):
        if i in feats_from:
            loss += reduce_fn(jnp.square(hr_feat - sr_feat), reduce)
    return loss


@register('losses', 'vgg')
class VGGLoss(Loss):
    def __init__(
            self, feats_from: Sequence[int], before_act: bool = False, reduce: str | Reduce = 'mean',
            weight: float = 1.
    ):
        super().__init__(reduce, weight)
        self.feats_from = feats_from
        self.before_act = before_act
        self.vgg_params = load_vgg19_params()

    def __call__(self, sr: jnp.ndarray, hr: jnp.ndarray) -> jnp.ndarray:
        return vgg_loss(sr, hr, self.vgg_params, self.feats_from, self.before_act, self.reduce) * self.weight


"""
Maybe Implement Style Loss in the future?
1. _gram_matrix
2. style_loss
3. StyleLoss
"""