import jax.numpy as jnp
from jax.random import PRNGKey

from flax.training import train_state
from flax import struct

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
    generator_losses: LossWrapper
    discriminator_losses: LossWrapper
    discriminator_tx: optax.GradientTransformation = struct.field(pytree_node=False)


def create_train_state(config: dict | DictConfig, init_key: PRNGKey) -> TrainState:
    model = get(module='models', **config['model'])
    inputs = jnp.ones((config['train_size']))
    params = model.init(init_key, inputs)

    generator_optimizer_kwargs = OmegaConf.to_container(config['generator_optimizer'], resolve=True)
    if not isinstance(generator_optimizer_kwargs['learning_rate'], float):
        generator_optimizer_kwargs['learning_rate'] = get(
            module='lr_schedules', **generator_optimizer_kwargs['learning_rate'])
    generator_tx = get(module='optimizers', **generator_optimizer_kwargs)
    generator_losses = [get(module='losses', **c) for c in config['generator_loss']] \
        if isinstance(config['generator_loss'], Sequence) else [get(module='losses', **config['generator_loss'])]
    generator_weights = config['generator_loss_weight'] if config.get('generator_loss_weight') is not None else 1.
    generator_loss_wrapper = LossWrapper(losses=generator_losses, weights=generator_weights)

    discriminator_optimizer_kwargs = OmegaConf.to_container(config['discriminator_optimizer'], resolve=True)
    if not isinstance(discriminator_optimizer_kwargs['learning_rate'], float):
        discriminator_optimizer_kwargs['learning_rate'] = get(
            module='lr_schedules', **discriminator_optimizer_kwargs['learning_rate'])
    discriminator_tx = get(module='optimizers', **discriminator_optimizer_kwargs)
    discriminator_losses = [get(module='losses', **c) for c in config['discriminator_loss']] \
        if isinstance(config['discriminator_loss'], Sequence) else [get(module='losses', **config['discriminator_loss'])]
    discriminator_weights = config['discriminator_loss_weight']\
        if config.get('discriminator_loss_weight') is not None else 1.
    discriminator_loss_wrapper = LossWrapper(losses=discriminator_losses, weights=discriminator_weights)

    t_s = TrainState.create(
        apply_fn=model.apply, params=params, tx=generator_tx, losses=generator_loss_wrapper,
        discriminator_tx=discriminator_tx, discriminator_losses=discriminator_loss_wrapper)
    return t_s
