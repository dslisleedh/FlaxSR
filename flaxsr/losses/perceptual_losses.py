import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

from flaxsr.losses.utils import reduce_fn, Reduces
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
        hr: jnp.ndarray, sr: jnp.ndarray, vgg_params: Pytree,
        feats_from: Sequence[int], before_act: bool = False, reduces: str | Reduces = 'mean'
) -> jnp.ndarray:
    hr_feats = _get_feats_from_vgg19(hr, vgg_params, before_act)
    sr_feats = _get_feats_from_vgg19(sr, vgg_params, before_act)

    loss = 0.
    for i, (hr_feats, sr_feats) in enumerate(zip(hr_feats, sr_feats)):
        if i in feats_from:
            loss += jnp.mean((hr_feats - sr_feats) ** 2, axis=(1, 2, 3))
    return reduce_fn(loss, reduces)


@register('losses', 'vgg')
class VGGLoss:
    def __init__(
            self, feats_from: Sequence[int], before_act: bool = False, reduces: str | Reduces = 'mean'
    ):
        self.feats_from = feats_from
        self.before_act = before_act
        self.reduces = reduces

    def __call__(self, hr: jnp.ndarray, sr: jnp.ndarray, vgg_params: Pytree) -> jnp.ndarray:
        return vgg_loss(hr, sr, vgg_params, self.feats_from, self.before_act, self.reduces)


"""
Maybe Implement Style Loss in the future?
1. _gram_matrix
2. style_loss
3. StyleLoss
"""