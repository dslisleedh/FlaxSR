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


def minmax_discriminator_loss(
        true: jnp.ndarray, fake: jnp.ndarray, reduces: str | Reduces = 'mean',
        from_logits: bool = True, *args, **kwargs
):
    if from_logits:
        true_loss = -jax.nn.log_sigmoid(true)
        fake_loss = -jax.nn.log_sigmoid(-fake)
    else:
        true_loss = -jnp.log(true)
        fake_loss = -jnp.log(1. - fake)

    loss = true_loss + fake_loss
    return reduce_fn(loss, reduces)


def minmax_generator_loss(
        fake: jnp.ndarray, reduces: str | Reduces = 'mean', from_logits: bool = True,
        *args, **kwargs
):
    if from_logits:
        loss = -jax.nn.log_sigmoid(fake)
    else:
        loss = -jnp.log(fake)

    return reduce_fn(loss, reduces)


@register('losses', 'minmax_discriminator')
class MinmaxDiscriminatorLoss:
    def __init__(self, from_logits: bool = True, reduces: str | Reduces = 'mean'):
        self.from_logits = from_logits
        self.reduces = reduces

    def __call__(self, true: jnp.ndarray, fake: jnp.ndarray):
        return minmax_discriminator_loss(true, fake, self.reduces, self.from_logits)


@register('losses', 'minmax_generator')
class MinmaxGeneratorLoss:
    def __init__(self, from_logits: bool = True, reduces: str | Reduces = 'mean'):
        self.from_logits = from_logits
        self.reduces = reduces

    def __call__(self, fake: jnp.ndarray):
        return minmax_generator_loss(fake, self.reduces, self.from_logits)


def least_square_discriminator_loss(
        true: jnp.ndarray, fake: jnp.ndarray, from_logits: bool = True, reduces: str | Reduces = 'mean',
        *args, **kwargs
):
    if from_logits:
        true = jax.nn.sigmoid(true)
        fake = jax.nn.sigmoid(fake)

    true_loss = .5 * jnp.square(true - 1.)
    fake_loss = .5 * jnp.square(fake)

    loss = true_loss + fake_loss
    return reduce_fn(loss, reduces)


def least_square_generator_loss(
        fake: jnp.ndarray, from_logits: bool = True, reduces: str | Reduces = 'mean',
        *args, **kwargs
):
    if from_logits:
        fake = jax.nn.sigmoid(fake)

    loss = .5 * jnp.square(fake - 1.)

    return reduce_fn(loss, reduces)


@register('losses', 'least_square_discriminator')
class LeastSquareDiscriminatorLoss:
    def __init__(self, from_logits: bool = True, reduces: str | Reduces = 'mean'):
        self.from_logits = from_logits
        self.reduces = reduces

    def __call__(self, true: jnp.ndarray, fake: jnp.ndarray):
        return least_square_discriminator_loss(true, fake, self.from_logits, self.reduces)


@register('losses', 'least_square_generator')
class LeastSquareGeneratorLoss:
    def __init__(self, from_logits: bool = True, reduces: str | Reduces = 'mean'):
        self.from_logits = from_logits
        self.reduces = reduces

    def __call__(self, fake: jnp.ndarray):
        return least_square_generator_loss(fake, self.from_logits, self.reduces)


def relativistic_discriminator_loss(
        true: jnp.ndarray, fake: jnp.ndarray, reduces: str | Reduces = 'mean',
        *args, **kwargs
):
    true_loss = -jax.nn.log_sigmoid(true - jnp.mean(fake))
    fake_loss = -jax.nn.log_sigmoid(-fake + jnp.mean(true))

    loss = true_loss + fake_loss
    return reduce_fn(loss, reduces)


def relativistic_generator_loss(
        true: jnp.ndarray, fake: jnp.ndarray, reduces: str | Reduces = 'mean',
        *args, **kwargs
):
    true_loss = -jax.nn.log_sigmoid(-true + jnp.mean(fake))
    fake_loss = -jax.nn.log_sigmoid(fake - jnp.mean(true))

    loss = true_loss + fake_loss
    return reduce_fn(loss, reduces)


@register('losses', 'relativistic_discriminator')
class RelativisticDiscriminatorLoss:
    def __init__(self, reduces: str | Reduces = 'mean'):
        self.reduces = reduces

    def __call__(self, true: jnp.ndarray, fake: jnp.ndarray):
        return relativistic_discriminator_loss(true, fake, self.reduces)


@register('losses', 'relativistic_generator')
class RelativisticGeneratorLoss:
    def __init__(self, reduces: str | Reduces = 'mean'):
        self.reduces = reduces

    def __call__(self, true: jnp.ndarray, fake: jnp.ndarray):
        return relativistic_generator_loss(true, fake, self.reduces)
