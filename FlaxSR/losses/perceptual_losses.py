import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

import pickle
import os

from FlaxSR.losses.utils import reduce_fn


"""
Not tested yet
"""


Pytree = any


def _get_package_dir():
    current_file = os.path.realpath(__file__)
    return os.path.dirname(current_file)


def check_vgg_params_exists():
    dir_path = _get_package_dir()
    state = os.path.exists(os.path.join(dir_path, 'vgg16_weights.pkl'))

    if not state:
        print('VGG19 weights not found !')
        print('Downloading VGG19 weights ...')
        import tensorflow as tf
        vgg16 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        weights = []
        for layer in vgg16.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                weights.append((jnp.asarray(layer.weights[0].numpy()), jnp.asarray(layer.weights[1].numpy())))
        with open(os.path.join(dir_path, 'vgg16_weights.pkl'), 'wb') as f:
            pickle.dump(weights, f)
        print('Done !')


def _conv(x: jnp.ndarray, weights: jnp.ndarray, bias: jnp.ndarray):
    x = lax.conv_general_dilated(
        x, weights, (1, 1), 'SAME', (1, 1), (1, 1), dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    x = x + bias
    return x


def _get_feats_from_vgg(x: jnp.ndarray, params: Pytree, feats_from: Sequence[int], before_act: bool):
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


def vgg_loss(hr: jnp.ndarray, sr: jnp.ndarray, feats_from: Sequence[int], before_act: bool):
    with open(os.path.join(_get_package_dir(), 'vgg16_weights.pkl'), 'rb') as f:
        params = pickle.load(f)

    hr_feats = _get_feats_from_vgg(hr, params, feats_from, before_act)
    sr_feats = _get_feats_from_vgg(sr, params, feats_from, before_act)

    loss = 0
    for hr_feat, sr_feat in zip(hr_feats, sr_feats):
        loss += jnp.mean((hr_feat - sr_feat) ** 2)

    return loss
