# FlaxSR

Super Resolution models with Jax/Flax

## HOW TO USE

```python
from FlaxSR.models import VDSR
import jax
import jax.numpy as jnp

inputs = jnp.ones((16, 256, 256, 3))
key = jax.random.PRNGKey(42)
model = VDSR(n_filters=64, n_blocks=20, scale=4)
params = model.init(key, inputs)
outputs = model.apply(params, inputs)
print(outputs.shape)
```

## Feats will be added

 - [ ] More models
 - [ ] Pre-trained parameters
 - [ ] Training states(includes Generative-sr models)
