import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

from flaxsr.losses.utils import reduce_fn, Reduce, Loss
from flaxsr._utils import register


@partial(jax.jit, static_argnums=(2, 3,))
def minmax_discriminator_loss(
        fake: jnp.ndarray, true: jnp.ndarray, from_logits: bool = True, reduce: str | Reduce = 'mean',
        *args, **kwargs
):
    if from_logits:
        true_loss = -jax.nn.log_sigmoid(true)
        fake_loss = -jax.nn.log_sigmoid(-fake)
    else:
        true_loss = -jnp.log(true)
        fake_loss = -jnp.log(1. - fake)

    loss = true_loss + fake_loss
    return reduce_fn(loss, reduce)


@partial(jax.jit, static_argnums=(1, 2,))
def minmax_generator_loss(
        fake: jnp.ndarray, from_logits: bool = True, reduce: str | Reduce = 'mean',
        *args, **kwargs
):
    if from_logits:
        loss = -jax.nn.log_sigmoid(fake)
    else:
        loss = -jnp.log(fake)

    return reduce_fn(loss, reduce)


@register('losses', 'minmax_discriminator')
class MinmaxDiscriminatorLoss(Loss):
    def __init__(self, from_logits: bool = True, reduce: str | Reduce = 'mean', weight: float = 1.):
        super().__init__(reduce, weight)
        self.from_logits = from_logits

    def __call__(self, fake: jnp.ndarray, true: jnp.ndarray, *args, **kwargs):
        return minmax_discriminator_loss(fake, true, self.from_logits, self.reduce) * self.weight


@register('losses', 'minmax_generator')
class MinmaxGeneratorLoss(Loss):
    def __init__(self, from_logits: bool = True, reduce: str | Reduce = 'mean', weight: float = 1.):
        super().__init__(reduce, weight)
        self.from_logits = from_logits

    def __call__(self, fake: jnp.ndarray, *args, **kwargs):
        return minmax_generator_loss(fake, self.from_logits, self.reduce) * self.weight


@partial(jax.jit, static_argnums=(2, 3,))
def least_square_discriminator_loss(
        fake: jnp.ndarray, true: jnp.ndarray, from_logits: bool = True, reduce: str | Reduce = 'mean',
        *args, **kwargs
):
    if from_logits:
        true = jax.nn.sigmoid(true)
        fake = jax.nn.sigmoid(fake)

    true_loss = .5 * jnp.square(true - 1.)
    fake_loss = .5 * jnp.square(fake)

    loss = true_loss + fake_loss
    return reduce_fn(loss, reduce)


@partial(jax.jit, static_argnums=(1, 2,))
def least_square_generator_loss(
        fake: jnp.ndarray, from_logits: bool = True, reduce: str | Reduce = 'mean',
        *args, **kwargs
):
    if from_logits:
        fake = jax.nn.sigmoid(fake)

    loss = .5 * jnp.square(fake - 1.)

    return reduce_fn(loss, reduce)


@register('losses', 'least_square_discriminator')
class LeastSquareDiscriminatorLoss(Loss):
    def __init__(self, from_logits: bool = True, reduce: str | Reduce = 'mean', weight: float = 1.):
        super().__init__(reduce, weight)
        self.from_logits = from_logits

    def __call__(self, fake: jnp.ndarray, true: jnp.ndarray, *args, **kwargs):
        return least_square_discriminator_loss(fake, true, self.from_logits, self.reduce) * self.weight


@register('losses', 'least_square_generator')
class LeastSquareGeneratorLoss(Loss):
    def __init__(self, from_logits: bool = True, reduce: str | Reduce = 'mean', weight: float = 1.):
        super().__init__(reduce, weight)
        self.from_logits = from_logits

    def __call__(self, fake: jnp.ndarray, *args, **kwargs):
        return least_square_generator_loss(fake, self.from_logits, self.reduce) * self.weight


def d_ra(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return x - jnp.mean(y)


@partial(jax.jit, static_argnums=(2,))
def relativistic_discriminator_loss(
        fake: jnp.ndarray, true: jnp.ndarray, reduce: str | Reduce = 'mean',
        *args, **kwargs
):
    true_loss = -jax.nn.log_sigmoid(d_ra(true, fake))
    fake_loss = -jnp.log(1. - jax.nn.sigmoid(d_ra(fake, true)))

    loss = true_loss + fake_loss
    return reduce_fn(loss, reduce)


@partial(jax.jit, static_argnums=(2,))
def relativistic_generator_loss(
        fake: jnp.ndarray, true: jnp.ndarray, reduce: str | Reduce = 'mean',
        *args, **kwargs
):
    true_loss = -jnp.log(1. - jax.nn.sigmoid(d_ra(true, fake)))
    fake_loss = -jax.nn.log_sigmoid(d_ra(fake, true))

    loss = true_loss + fake_loss
    return reduce_fn(loss, reduce)


@register('losses', 'relativistic_discriminator')
class RelativisticDiscriminatorLoss(Loss):
    def __init__(self, reduce: str | Reduce = 'mean', weight: float = 1.):
        super().__init__(reduce, weight)

    def __call__(self, fake: jnp.ndarray, true: jnp.ndarray, *args, **kwargs):
        return relativistic_discriminator_loss(fake, true, self.reduce) * self.weight


@register('losses', 'relativistic_generator')
class RelativisticGeneratorLoss(Loss):
    def __init__(self, reduce: str | Reduce = 'mean', weight: float = 1.):
        super().__init__(reduce, weight)

    def __call__(self, fake: jnp.ndarray, true: jnp.ndarray, *args, **kwargs):
        return relativistic_generator_loss(fake, true, self.reduce) * self.weight
