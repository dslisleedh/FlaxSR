import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

from flaxsr.losses.utils import reduce_fn
from flaxsr._utils import register


"""
Not tested yet
"""


Pytree = any


def _conv(x: jnp.ndarray, weights: jnp.ndarray, bias: jnp.ndarray):
    x = lax.conv_general_dilated(
        x, weights, (1, 1), 'SAME', (1, 1), (1, 1), dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    x = x + bias
    return x


def _get_feats_from_vgg19(x: jnp.ndarray, params: Pytree, feats_from: Sequence[int], before_act: bool):
    max_pool_indices = [2, 4, 8, 12]

    outputs = []
    for i, (w, b) in enumerate(params):
        if i in max_pool_indices:
            x = nn.max_pool(x, (2, 2), (2, 2), 'SAME')
        x = _conv(x, w, b)
        if i in feats_from:
            if before_act:
                outputs.append(x)
                x = nn.relu(x)
            else:
                x = nn.relu(x)
                outputs.append(x)
        else:
            x = nn.relu(x)
    return outputs


@register('losses', 'vgg')
def vgg_loss(
        hr: jnp.ndarray, sr: jnp.ndarray, vgg_params: Pytree,
        feats_from: Sequence[int], before_act: bool = False, mode: Literal['mean', 'sum', None] = 'mean'
):
    hr_feats = _get_feats_from_vgg19(hr, vgg_params, feats_from, before_act)
    sr_feats = _get_feats_from_vgg19(sr, vgg_params, feats_from, before_act)

    loss = 0
    for hr_feat, sr_feat in zip(hr_feats, sr_feats):
        loss += jnp.mean((hr_feat - sr_feat) ** 2, axis=(1, 2, 3))

    return reduce_fn(loss, mode)
