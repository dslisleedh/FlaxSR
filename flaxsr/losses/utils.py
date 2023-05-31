import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal, Any, Optional
from enum import Enum

import os
import pickle

from flaxsr._utils import get


loss_wrapper = Sequence[tuple[Any, float]]


class Reduces(Enum):
    SUM = 'sum'
    MEAN = 'mean'
    NONE = 'none'


def reduce_fn(loss, reduce: str | Reduces) -> jnp.ndarray:
    if reduce == Reduces.SUM:
        return jnp.sum(loss)
    elif reduce == Reduces.MEAN:
        return jnp.mean(loss)
    elif reduce == Reduces.NONE:
        return loss
    else:
        raise ValueError(f"Unknown reduce type {reduce}")


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
                weights.append(
                    (
                        jnp.asarray(layer.weights[0].numpy()),
                        jnp.asarray(layer.weights[1].numpy())[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...])
                )
        with open(os.path.join(dir_path, 'vgg19_weights.pkl'), 'wb') as f:
            pickle.dump(weights, f)
        print('Done !')


def load_vgg19_params():
    dir_path = _get_package_dir()
    params = pickle.load(open(os.path.join(dir_path, 'vgg19_weights.pkl'), 'rb'))
    return params


def get_loss_wrapper(
        losses: Sequence[str], weights: Sequence[float], modes: str | Reduces | Sequence[str | Reduces] = 'mean'
) -> loss_wrapper:
    assert len(losses) == len(weights), \
        f"Number of losses and weights must be equal, got {len(losses)} and {len(weights)}"
    if isinstance(modes, str | Reduces):
        modes = [modes] * len(losses)
    return [(get('losses', loss, reduce=mode), float(weight)) for loss, weight, mode in zip(losses, weights, modes)]


def compute_loss(
        hr: jnp.ndarray, sr: jnp.ndarray, losses: loss_wrapper, mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    loss = jnp.zeros(())

    if mask is not None:
        hr = hr * mask
        sr = sr * mask

    for loss_fn, weight in losses:
        loss += weight * loss_fn(hr, sr)

    return loss
