import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

from flaxsr._utils import register
from flaxsr.training.discriminative.train_state import TrainState


# @register('train_step', 'discriminative')
def train_step(state: TrainState, batch: Sequence[jnp.ndarray]) -> tuple[TrainState, jnp.ndarray]:
    if len(batch) == 2:
        lr, hr = batch
        mask = None
    else:
        lr, hr, mask = batch

    def loss_fn(params):
        _sr = state.apply_fn(params, lr)
        _loss = state.losses(_sr, hr, mask)
        return _loss, _sr

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, sr), grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return state, loss
