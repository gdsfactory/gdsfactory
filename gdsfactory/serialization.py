"""Serialize component settings into YAML or strings."""

import copy
import functools
import hashlib
import inspect
import pathlib
from typing import Any, Dict

import numpy as np
import orjson
import pydantic
import toolz
from omegaconf import DictConfig, OmegaConf
from phidl.device_layout import Path as PathPhidl


def clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans dictionary recursively."""
    for k, v in d.items():
        d[k] = clean_dict(dict(v)) if isinstance(v, dict) else clean_value_json(v)
    return d


def get_string(value: Any) -> str:
    try:
        s = orjson.dumps(
            value, option=orjson.OPT_SERIALIZE_NUMPY, default=clean_value_name
        ).decode()
    except TypeError as e:
        print(f"Error serializing {value!r}")
        raise e
    return s


def clean_value_json(value: Any) -> Any:
    """Return JSON serializable object."""
    if isinstance(value, pydantic.BaseModel):
        return value.dict()

    elif isinstance(value, bool):
        return value

    elif isinstance(value, (np.integer, int)):
        return int(value)

    elif isinstance(value, (float, np.inexact, np.float64)):
        return float(value)

    elif isinstance(value, np.ndarray):
        value = np.round(value, 3)
        return orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY).decode()
    elif callable(value) and isinstance(value, functools.partial):
        sig = inspect.signature(value.func)
        args_as_kwargs = dict(zip(sig.parameters.keys(), value.args))
        args_as_kwargs.update(**value.keywords)
        clean_dict(args_as_kwargs)
        args_as_kwargs.pop("function", None)

        func = value.func
        while hasattr(func, "func"):
            func = func.func
        return dict(function=func.__name__, settings=args_as_kwargs)

    elif hasattr(value, "to_dict"):
        return value.to_dict()
    elif callable(value) and isinstance(value, toolz.functoolz.Compose):
        value = [clean_value_json(value.first)] + [
            clean_value_json(func) for func in value.funcs
        ]
    elif callable(value) and hasattr(value, "__name__"):
        value = dict(function=value.__name__)
    elif isinstance(value, PathPhidl):
        value = value.hash_geometry()
    elif isinstance(value, pathlib.Path):
        value = value.stem
    elif isinstance(value, dict):
        value = copy.deepcopy(value)
        value = clean_dict(value)
    elif isinstance(value, DictConfig):
        value = clean_dict(OmegaConf.to_container(value))
    else:
        try:
            value_json = orjson.dumps(
                value, option=orjson.OPT_SERIALIZE_NUMPY, default=clean_value_json
            )
            value = orjson.loads(value_json)
        except TypeError as e:
            print(f"Error serializing {value!r}")
            raise e
    return value

    # elif isinstance(value, (tuple, list, ListConfig)):
    #     value = [clean_value_json(i) for i in value]
    # elif value is None:
    #     value = None
    # elif hasattr(value, "name"):
    #     value = value.name
    # elif hasattr(value, "get_name"):
    #     value = value.get_name()
    # else:
    #     value = str(value)


def clean_value_name(value: Any) -> str:
    """Returns a string representation of an object."""
    # value1 = clean_value_json(value)
    # print(type(value), value, value1, str(value1))
    return str(clean_value_json(value))


def get_hash(value: Any) -> str:
    return hashlib.md5((clean_value_name(value)).encode()).hexdigest()[:8]


if __name__ == "__main__":
    import gdsfactory as gf

    # f = gf.partial(gf.c.straight, length=3)
    # d = clean_value_json(f)
    # print(f"{d!r}")

    f = gf.partial(gf.c.straight, length=3)
    c = f()
    d = clean_value_json(c)
    print(d, d)
