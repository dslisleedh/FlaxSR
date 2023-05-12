import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal, Any

import os
import pickle


loss_wrapper = Any


def reduce_fn(loss, reduce: Literal['sum', 'mean', None]) -> jnp.ndarray:
    if reduce == 'sum':
        return jnp.sum(loss)
    elif reduce == 'mean':
        return jnp.mean(loss)
    elif reduce is None:
        return loss
    else:
        raise ValueError(f"Invalid reduce mode, got {reduce}. Must be ['sum', 'mean', None]")


def _get_package_dir():
    current_file = os.path.realpath(__file__)
    return os.path.dirname(current_file)


def check_vgg_params_exists():
    dir_path = _get_package_dir()
    state = os.path.exists(os.path.join(dir_path, 'vgg19_weights.pkl'))

    if not state:
        print('VGG19 weights not found !!!')
        print('Downloading VGG19 weights ...')
        import tensorflow as tf
        vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        weights = []
        for layer in vgg19.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                weights.append((jnp.asarray(layer.weights[0].numpy()), jnp.asarray(layer.weights[1].numpy())))
        with open(os.path.join(dir_path, 'vgg19_weights.pkl'), 'wb') as f:
            pickle.dump(weights, f)
        print('Done !')


def load_vgg19_params():
    dir_path = _get_package_dir()
    params = pickle.load(open(os.path.join(dir_path, 'vgg19_weights.pkl'), 'rb'))
    return params


def get_loss_wrapper(losses: Sequence, weights: Sequence) -> loss_wrapper:
    assert len(losses) == len(weights), \
        f"Number of losses and weights must be equal, got {len(losses)} and {len(weights)}"
    return [(loss, weight) for loss, weight in zip(losses, weights)]


def compute_loss(
        hr: jnp.ndarray, sr: jnp.ndarray, losses: loss_wrapper, mode: Literal['sum', 'mean', None] = 'sum'
) -> jnp.ndarray:
    loss = 0.
    for loss_fn, weight in losses:
        loss = loss + (reduce_fn(loss_fn(hr, sr), mode) * weight)
    return loss
