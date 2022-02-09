import copy
import functools
import inspect
from typing import Any, Dict

import numpy as np
import orjson
import pydantic
import toolz
from dotmap import DotMap
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from phidl.device_layout import Path as PathPhidl

from gdsfactory.cross_section import CrossSection
from gdsfactory.hash_points import hash_points
from gdsfactory.port import Port


def clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans dictionary keys recursively."""
    for k, v in d.items():
        if isinstance(v, DotMap):
            d[k] = dict(v)
        elif isinstance(v, dict):
            d[k] = clean_dict(v)
        else:
            d[k] = clean_value_json(v)
    return d


def clean_key(key):
    if isinstance(key, tuple):
        key = key[0]
    else:
        key = str(key)

    return key


def clean_value_json(value: Any) -> Any:
    """Return JSON serializable object."""
    if isinstance(value, pydantic.BaseModel):
        value = dict(value)
    elif isinstance(value, Port):
        value = value.to_dict()
    elif isinstance(value, float) and int(value) == value:
        value = int(value)
    elif isinstance(value, (np.int64, np.int32)):
        value = int(value)
    elif isinstance(value, np.ndarray):
        value = np.round(value, 3)
        value = orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY)
    elif isinstance(value, np.float64):
        value = float(value)
    elif type(value) in [int, float, str, bool]:
        pass
    elif callable(value) and isinstance(value, toolz.functoolz.Compose):
        value = [clean_value_json(value.first)] + [
            clean_value_json(func) for func in value.funcs
        ]
    elif callable(value) and hasattr(value, "__name__"):
        value = dict(function=value.__name__)
    elif callable(value) and isinstance(value, functools.partial):
        sig = inspect.signature(value.func)
        args_as_kwargs = dict(zip(sig.parameters.keys(), value.args))
        args_as_kwargs.update(**value.keywords)
        clean_dict(args_as_kwargs)
        args_as_kwargs.pop("function", None)
        value = dict(function=value.func.__name__, **args_as_kwargs)

    elif isinstance(value, DotMap):
        value = dict(value)
    elif isinstance(value, CrossSection):
        # value = value.info
        value = value.to_dict()
        value = copy.deepcopy(value)
        value = clean_dict(value)
    elif isinstance(value, Port):
        value = copy.deepcopy(value.settings)
        value = clean_dict(value)
    elif isinstance(value, dict):
        value = copy.deepcopy(value)
        value = clean_dict(value)
    elif isinstance(value, DictConfig):
        clean_dict(value)
    elif isinstance(value, PathPhidl):
        value = f"path_{hash_points(value.points)}"
    elif isinstance(value, (tuple, list, ListConfig)):
        value = [clean_value_json(i) for i in value]
    elif value is None:
        value = None
    elif hasattr(value, "name"):
        value = value.name
    elif hasattr(value, "get_name"):
        value = value.get_name()
    else:
        value = str(value)

    return value


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.c.straight()

    d = clean_value_json(c.ports)
