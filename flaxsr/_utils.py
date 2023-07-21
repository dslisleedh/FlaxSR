import optax
import copy


"""
Maybe fix this later to load model of specific size by name
"""


module_dicts = {}
optimizers = {
    'adabelief': optax.adabelief,
    'adafactor': optax.adafactor,
    'adagrad': optax.adagrad,
    'adam': optax.adam,
    'adamw': optax.adamw,
    'adamax': optax.adamax,
    'adamaxw': optax.adamaxw,
    'amsgrad': optax.amsgrad,
    'fromage': optax.fromage,
    'lamb': optax.lamb,
    'lars': optax.lars,
    'lion': optax.lion,
    'noisy_sgd': optax.noisy_sgd,
    'novograd': optax.novograd,
    'optimistic_gradient_descent': optax.optimistic_gradient_descent,
    'dpsgd': optax.dpsgd,
    'radam': optax.radam,
    'rmsprop': optax.rmsprop,
    'sgd': optax.sgd,
    'sm3': optax.sm3,
    'yogi': optax.yogi
}
lr_schedules = {
    'constant': optax.constant_schedule,
    'cosine_decay': optax.cosine_decay_schedule,
    'cosine_onecycle_schedule': optax.cosine_onecycle_schedule,
    'exponential_decay': optax.exponential_decay,
    'join_schedules': optax.join_schedules,
    'linear_onecycle_schedule': optax.linear_onecycle_schedule,
    'linear_schedule': optax.linear_schedule,
    'piecewise_constant_schedule': optax.piecewise_constant_schedule,
    'piecewise_interpolate_schedule': optax.piecewise_interpolate_schedule,
    'polynomial_schedule': optax.polynomial_schedule,
    'sgdr_schedule': optax.sgdr_schedule,
    'warmup_cosine_decay_schedule': optax.warmup_cosine_decay_schedule,
    'warmup_exponential_decay_schedule': optax.warmup_exponential_decay_schedule,
    'inject_hyperparams': optax.inject_hyperparams
}
module_dicts['optimizers'] = optimizers
module_dicts['lr_schedules'] = lr_schedules


def register(module: str, name: str):
    def decorator(cls):
        if module not in module_dicts:
            module_dicts[module] = {}
        module_dicts[module][name] = cls
        return cls
    return decorator


def get(module: str, name: str, *args, **kwargs):
    if module not in module_dicts:
        raise ValueError(f"Module {module} not found")
    if name not in module_dicts[module]:
        raise ValueError(f"Name {name} not found in module {module}")
    cls = copy.deepcopy(module_dicts[module][name])
    return cls(*args, **kwargs)
