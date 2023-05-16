import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn
from flaxsr._utils import register

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal


@register('layers', 'droppath')
class DropPath(nn.Module):
    survival_prob: float

    @nn.compact
    def __call__(self, skip: jnp.ndarray, residual: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if training:
            if self.survival_prob == 0:
                return skip

            key = self.make_rng('DropPath')
            shape = [s if i == 0 else 1 for i, s in enumerate(skip.shape)]
            survival_state = jax.random.bernoulli(key, self.survival_prob, shape)
            return jnp.where(survival_state, (skip + residual) / self.survival_prob, skip)

        else:
            return skip + residual


@register('layers', 'droppath_fast')
class DropPathFast(nn.Module):
    module: nn.Module
    survival_prob: float

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if not training or self.survival_prob == 1:
            return self.module(inputs)

        else:
            if self.survival_prob == 0:
                return inputs

            else:
                key = self.make_rng('DropPath')
                survival_state = jax.random.bernoulli(key, self.survival_prob, ())
                if survival_state:
                    residual = (self.module(inputs) - inputs) / self.survival_prob
                else:
                    residual = jnp.zeros_like(inputs)

                return inputs + residual
