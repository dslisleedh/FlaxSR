# FlaxSR

Super Resolution models with Jax/Flax

## HOW TO USE

### Install
```shell
pip install flaxsr
```

### Usage
 - Models
```python
import flaxsr
from flaxsr.models import VDSR
import jax
import jax.numpy as jnp
import numpy as np

inputs = jnp.ones((16, 256, 256, 3))
key = jax.random.PRNGKey(42)
model = VDSR(n_filters=64, n_blocks=20, scale=4)
params = model.init(key, inputs)
outputs = model.apply(params, inputs)
print(outputs.shape)

# Or you can use flaxsr.get function
model_get = flaxsr.get("models", "vdsr", n_filters=64, n_blocks=20, scale=4)
params_get = model_get.init(key, inputs)
outputs_get = model_get.apply(params_get, inputs)
print(outputs_get.shape)

np.allclose(outputs, outputs_get)
```

 - Losses
```python
import flaxsr
import jax
import jax.numpy as jnp
import numpy as np

hr = jnp.asarray(np.random.normal((16, 256, 256, 3)))
sr = jnp.asarray(np.random.normal((16, 256, 256, 3)))

loss = flaxsr.losses.l1_loss(hr, sr, "mean")

# Or you can use flaxsr.get function
metric = flaxsr.get("losses", "l1", "mean")
loss_get = metric(hr, sr)

np.allclose(loss, loss_get)
```

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
model = flaxsr.get("models", "vdsr", **model_kwargs)
losses = flaxsr.losses.LossWrapper(
    losses=['l1', 'l2'], weights=[1.0, 1.0]
)
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 8, 8, 3), dtype=jnp.float32))
tx = optax.adam(1e-3)

state = flaxsr.training.TrainState.create(
    apply_fn=model.apply, params=params, tx=tx, losses=losses
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
 - NAFSSR
