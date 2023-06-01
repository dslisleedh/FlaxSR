import jax
import jax.lax as lax
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.training.train_state import TrainState

import numpy as np
import einops

from functools import partial
from typing import Sequence, Literal

from flaxsr._utils import register



@register('train_step', 'discriminative')
def discriminative_train_step(state: TrainState, batch: Sequence[jnp.ndarray]) -> Sequence:
    if len(batch) == 2:
        lr, hr = batch
    else:
        lr, hr, mask = batch

    def loss_fn(params):
        _sr = state.apply_fn(params, lr)
        _loss = state.losses(hr, _sr, mask if len(batch) == 3 else None)
        return _loss, _sr

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, sr), grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return state, loss


# @register('train_step', 'generative')
# def generative_train_step(state: TrainState, batch: Sequence[jnp.ndarray]) -> Sequence:
#     if len(batch) == 2:
#         lr, hr = batch
#     else:
#         lr, hr, mask = batch
#
#     def loss_fn(params):
#         _sr = state.apply_fn(params, lr)
#         _loss = compute_loss(hr, _sr, state.losses, state.reduce, mask if len(batch) == 3 else None)
#         return _loss, _sr
#
#     state = state.apply_gradients(grads=grad)
#     return state, loss
