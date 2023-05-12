import copy


"""
Maybe fix this later to load model of specific size by name
"""


module_dicts = {}


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
