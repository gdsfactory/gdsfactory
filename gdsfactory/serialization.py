"""Serialize component settings into YAML or strings."""
from __future__ import annotations

import functools
import hashlib
import inspect
import pathlib
from collections.abc import KeysView as dict_keys
from functools import partial
from typing import Any

import gdstk
import numpy as np
import orjson
import pydantic
import toolz
from omegaconf import DictConfig, OmegaConf

DEFAULT_SERIALIZATION_MAX_DIGITS = 3
"""By default, the maximum number of digits retained when serializing float-like arrays"""


def get_string(value: Any) -> str:
    try:
        s = orjson.dumps(
            value, option=orjson.OPT_SERIALIZE_NUMPY, default=clean_value_name
        ).decode()
    except TypeError as e:
        print(f"Error serializing {value!r}")
        raise e
    return s


def clean_dict(dictionary: dict) -> dict:
    return {k: clean_value_json(v) for k, v in dictionary.items()}


def complex_encoder(obj, digits=DEFAULT_SERIALIZATION_MAX_DIGITS):
    real_part = np.round(obj.real, digits)
    imag_part = np.round(obj.imag, digits)
    return {"real": real_part, "imag": imag_part}


def clean_value_json(value: Any) -> str | int | float | dict | list | bool | None:
    """Return JSON serializable object."""
    from gdsfactory.path import Path
    from gdsfactory.pdk import get_active_pdk

    active_pdk = get_active_pdk()
    include_module = active_pdk.cell_decorator_settings.include_module

    if isinstance(value, pydantic.BaseModel):
        return clean_dict(value.model_dump())

    elif hasattr(value, "get_component_spec"):
        return value.get_component_spec()

    elif isinstance(value, bool):
        return value

    elif isinstance(value, np.integer | int):
        return int(value)

    elif isinstance(value, float | np.inexact | np.float64):
        return float(np.round(value, DEFAULT_SERIALIZATION_MAX_DIGITS))

    elif isinstance(value, complex | np.complex64 | np.complex128):
        return complex_encoder(value)

    elif isinstance(value, np.ndarray):
        value = np.round(value, DEFAULT_SERIALIZATION_MAX_DIGITS)
        return orjson.loads(orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY))

    elif callable(value) and isinstance(value, functools.partial):
        sig = inspect.signature(value.func)
        args_as_kwargs = dict(zip(sig.parameters.keys(), value.args))
        args_as_kwargs.update(value.keywords)
        args_as_kwargs = clean_dict(args_as_kwargs)

        func = value.func
        while hasattr(func, "func"):
            func = func.func
        v = {
            "function": func.__name__,
            "settings": args_as_kwargs,
        }
        if include_module:
            v.update(module=func.__module__)
        return v

    elif hasattr(value, "to_dict"):
        return clean_dict(value.to_dict())

    elif callable(value) and isinstance(value, toolz.functoolz.Compose):
        return [clean_value_json(value.first)] + [
            clean_value_json(func) for func in value.funcs
        ]

    elif callable(value) and hasattr(value, "__name__"):
        return (
            {"function": value.__name__, "module": value.__module__}
            if include_module
            else {"function": value.__name__}
        )

    elif isinstance(value, Path):
        return value.hash_geometry()

    elif isinstance(value, pathlib.Path):
        return value.stem

    elif isinstance(value, dict):
        return clean_dict(value.copy())

    elif isinstance(value, DictConfig):
        return clean_dict(OmegaConf.to_container(value))

    elif isinstance(value, list | tuple | set | dict_keys):
        return [clean_value_json(i) for i in value]

    elif isinstance(value, gdstk.Polygon):
        return np.round(value.points, DEFAULT_SERIALIZATION_MAX_DIGITS)

    else:
        try:
            value_json = orjson.dumps(
                value, option=orjson.OPT_SERIALIZE_NUMPY, default=clean_value_json
            )
            return orjson.loads(value_json)
        except TypeError as e:
            print(f"Error serializing {value!r}")
            raise e


def clean_value_name(value: Any) -> str:
    """Returns a string representation of an object."""
    # value1 = clean_value_json(value)
    return str(clean_value_json(value))


def get_hash(value: Any) -> str:
    return hashlib.md5((clean_value_name(value)).encode()).hexdigest()[:8]


if __name__ == "__main__":
    import gdsfactory as gf

    # f = partial(gf.c.straight, length=3)
    # d = clean_value_json(f)
    # print(f"{d!r}")
    # f = partial(gf.c.straight, length=3)
    # c = f()
    # d = clean_value_json(c)
    # print(d, d)

    xs = partial(
        gf.cross_section.strip,
        width=3,
        add_pins=gf.partial(gf.add_pins.add_pins_inside1nm, pin_length=0.1),
    )
    f = partial(gf.routing.add_fiber_array, cross_section=xs)
    c = f()
    d = clean_value_json(c)
    print(get_hash(d))
    print(d, d)
