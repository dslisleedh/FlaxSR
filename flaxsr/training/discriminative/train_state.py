import jax.numpy as jnp
from jax.random import PRNGKey

from flax.training import train_state

from flaxsr._utils import register, get
from flaxsr.losses.utils import Loss, LossWrapper

import optax

from typing import Optional, Sequence
from omegaconf import DictConfig, OmegaConf


class TrainState(train_state.TrainState):
    """
    Custom TrainState for the SR model.
    Include losses inside to be able to use them in the training loop.
    """
    losses: LossWrapper | Loss


def create_train_state(config: dict | DictConfig, init_key: PRNGKey) -> TrainState:
    model = get(module='models', **config['model'])
    inputs = jnp.ones((config['train_size']))
    params = model.init(init_key, inputs)

    optimizer_kwargs = OmegaConf.to_container(config['optimizer'], resolve=True)
    if not isinstance(optimizer_kwargs['learning_rate'], float):
        optimizer_kwargs['learning_rate'] = get(module='lr_schedules', **optimizer_kwargs['learning_rate'])
    tx = get(module='optimizers', **optimizer_kwargs)

    losses = [get(module='losses', **c) for c in config['loss']] \
        if isinstance(config['loss'], Sequence) else [get(module='losses', **config['loss'])]
    weights = config['loss_weight'] if config.get('loss_weight') is not None else 1.
    loss_wrapper = LossWrapper(losses=losses, weights=weights)

    t_s = TrainState.create(apply_fn=model.apply, params=params, tx=tx, losses=loss_wrapper)
    return t_s
