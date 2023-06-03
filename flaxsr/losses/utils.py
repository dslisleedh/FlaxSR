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
from abc import ABC, abstractmethod

import os
import pickle

from flaxsr._utils import get


class Reduce(Enum):
    SUM = 'sum'
    MEAN = 'mean'
    NONE = 'none'


class Loss(ABC):
    def __init__(self, reduce: str | Reduce = 'mean'):
        self.reduce = reduce

    @abstractmethod
    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        """
        hr and sr for discriminative manner
        true and fake or only fake for generative manner
        """
        pass


def reduce_fn(loss, reduce: str | Reduce) -> jnp.ndarray:
    if isinstance(reduce, str):
        reduce = Reduce(reduce)

    reduce_axis = tuple(range(1, len(loss.shape)))

    if reduce == Reduce.SUM:
        stat = jnp.sum(loss, axis=reduce_axis)
    elif reduce == Reduce.MEAN:
        stat = jnp.mean(loss, axis=reduce_axis)
    elif reduce == Reduce.NONE:
        return loss
    else:
        raise ValueError(f"Unknown reduce type {reduce}")

    stat = jnp.mean(stat)
    return stat


def apply_mask(*args, mask: Optional[jnp.ndarray]) -> tuple[jnp.ndarray, ...]:
    if mask is None:
        return args
    return tuple(arg * mask for arg in args)


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


class LossWrapper:
    def __init__(self, losses: Sequence[Loss], weights: Sequence[float]):
        # Prevent user from using None with others(Sum, Mean)
        reduces = [loss.reduce for loss in losses]
        assert all(True if reduce in ['none', Reduce.NONE] else False for reduce in reduces) or \
               all(True if reduce not in ['none', Reduce.NONE] else False for reduce in reduces), \
            f'Cannot use None with others(Sum, Mean), got {reduces}'

        self.losses = losses
        self.weights = weights

    def __call__(self, sr: jnp.ndarray, hr: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        loss = jnp.zeros(())

        for loss_fn, weight in zip(self.losses, self.weights):
            loss += weight * loss_fn(sr, hr, mask)

        return loss
