# FlaxSR

Super Resolution models with Jax/Flax

## HOW TO USE

### Install
```shell
pip install flaxsr
```

### Usage
<b> You can easily load model/losses and train model using custom train_states. </b>

 - Train example
```python
import flaxsr
import jax
import jax.numpy as jnp
import numpy as np
import optax

model_kwargs = {
    'n_filters': 64, 'n_blocks': 8, 'scale': 4
}
model = flaxsr.get("models", "vdsr", **model_kwargs)  # This equals flaxsr.models.VDSR(**model_kwargs)
losses = [
    flaxsr.losses.L1Loss(reduce='sum'),
    flaxsr.get('losses', 'vgg', feats_from=(6, 8, 14,), before_act=False, reduce='mean')
]
loss_weights = (.1, 1.)
loss_wrapper = flaxsr.losses.LossWrapper(losses, loss_weights)
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 8, 8, 3), dtype=jnp.float32))
tx = optax.adam(1e-3)

state = flaxsr.training.TrainState.create(
    apply_fn=model.apply, params=params, tx=tx, losses=loss_wrapper
)

hr = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
lr = jnp.ones((1, 8, 8, 3), dtype=jnp.float32)
batch = (lr, hr)

state_new, loss = flaxsr.training.discriminative_train_step(state, batch)

assert state_new.step == 1
np.not_equal(state_new.params['params']['Conv_0']['kernel'], state.params['params']['Conv_0']['kernel'])
```


## Models implemented
 - SRCNN
 - FSRCNN
 - ESPCN
 - VDSR
 - EDSR, MDSR,
 - NCNet
 - SRResNet(SRGAN will be implemented in future)
