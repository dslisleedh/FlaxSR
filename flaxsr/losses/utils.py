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
    """
    Reduces exists to get statistics from spatial dimensions
    So, losses not calculated by pixel-wises like perceptual loss
    or not having spatial dimensions like total variation, This enum is not needed and has no effect
    """
    SUM = 'sum'
    MEAN = 'mean'
    NONE = 'none'


class Loss(ABC):
    def __init__(self, reduce: str | Reduce = Reduce.MEAN, weight: float = 1.):
        self.reduce = reduce if isinstance(reduce, Reduce) else Reduce(reduce)
        self.weight = weight

    @abstractmethod
    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        """
        hr and sr for discriminative manner
        true and fake or only fake for generative manner
        """
        pass


def reduce_fn(loss, reduce: str | Reduce) -> jnp.ndarray:
    reduce_axis = tuple(range(1, len(loss.shape)))  # (1, 2, 3) for (B, H, W, C). Exclude batch dim

    if reduce == Reduce.SUM:
        stat = jnp.sum(loss, axis=reduce_axis)
    elif reduce == Reduce.MEAN:
        stat = jnp.mean(loss, axis=reduce_axis)
    elif reduce == Reduce.NONE:
        return loss
    else:
        raise ValueError(f"Unknown reduce type {reduce}")

    stat = jnp.mean(stat)  # Reduce batch dim
    return stat


def _get_package_dir() -> str:
    current_file = os.path.realpath(__file__)
    return os.path.dirname(current_file)


def check_vgg_params_exists():
    dir_path = _get_package_dir()
    state = os.path.exists(os.path.join(dir_path, 'vgg19_weights.pkl'))

    if not state:
        print('VGG19 weights not found !!!')
        print('Downloading VGG19 weights ...')
        # To prevent errors when using jax and tensorflow together
        a = jnp.ones((1,))
        del a
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
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


def load_vgg19_params() -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    dir_path = _get_package_dir()
    params = pickle.load(open(os.path.join(dir_path, 'vgg19_weights.pkl'), 'rb'))
    return params


class LossWrapper:
    def __init__(self, losses: Sequence[Loss]):
        # Prevent user from using None with others(Sum, Mean)
        reduces = [loss.reduce for loss in losses]
        assert all(True if reduce in ['none', Reduce.NONE] else False for reduce in reduces) or \
               all(True if reduce not in ['none', Reduce.NONE] else False for reduce in reduces), \
            f'Cannot use None with others(Sum, Mean), got {reduces}'

        self.losses = losses

    def __call__(self, sr: jnp.ndarray, hr: jnp.ndarray) -> jnp.ndarray:
        loss = jnp.zeros(())

        for loss_fn in self.losses:
            loss += loss_fn(sr, hr)

        return loss
