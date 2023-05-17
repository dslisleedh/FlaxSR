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
metric = flaxsr.get("losses", "l1")
loss_get = metric(hr, sr, "mean")

np.allclose(loss, loss_get)
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
