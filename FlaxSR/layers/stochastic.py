import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal


class DropPath(nn.Module):
    survival_prob: float

    @nn.compact
    def __call__(self, skip: jnp.ndarray, residual: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if training:
            if self.survival_prob == 0:
                return skip

            key = self.make_rng('DropPath')
            survival_state = jax.random.bernoulli(key, self.survival_prob, skip.shape)
            return jnp.where(survival_state, (skip + residual) / self.survival_prob, skip)

        else:
            return skip + residual


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

