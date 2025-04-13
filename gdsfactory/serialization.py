"""Serialize component settings into YAML or strings."""

from __future__ import annotations

import functools
import hashlib
import inspect
import pathlib
import re
from collections.abc import KeysView
from keyword import iskeyword
from typing import Any, overload

import attrs
import numpy as np
import orjson
import pydantic
import toolz
from aenum import Enum

DEFAULT_SERIALIZATION_MAX_DIGITS = 3
"""By default, the maximum number of digits retained when serializing float-like arrays"""


def convert_tuples_to_lists(
    data: dict[str, Any] | list[Any] | tuple[Any, ...] | Any,
) -> dict[str, Any] | list[Any] | Any:
    if isinstance(data, dict):
        return {key: convert_tuples_to_lists(value) for key, value in data.items()}
    elif isinstance(data, list | tuple):
        return [convert_tuples_to_lists(item) for item in data]
    return data


def get_string(value: Any) -> str:
    try:
        s = orjson.dumps(
            value, option=orjson.OPT_SERIALIZE_NUMPY, default=clean_value_name
        ).decode()
    except TypeError as e:
        print(f"Error serializing {value!r}")
        raise e
    return str(s)


def clean_dict(dictionary: dict[str, Any]) -> dict[str, Any]:
    return {k: clean_value_json(v) for k, v in dictionary.items()}


def complex_encoder(
    obj: complex | np.complexfloating, digits: int = DEFAULT_SERIALIZATION_MAX_DIGITS
) -> dict[str, Any]:
    real_part = np.round(obj.real, digits)
    imag_part = np.round(obj.imag, digits)
    return {"real": real_part, "imag": imag_part}


@overload
def clean_value_json(
    value: dict[str, Any],
    include_module: bool = True,
    serialize_function_as_dict: bool = True,
) -> dict[str, Any]: ...


@overload
def clean_value_json(
    value: str,
    include_module: bool = True,
    serialize_function_as_dict: bool = True,
) -> str: ...


@overload
def clean_value_json(
    value: Any,
    include_module: bool = True,
    serialize_function_as_dict: bool = True,
) -> str | int | float | dict[str, Any] | list[Any] | bool | Any | None: ...


def clean_value_json(
    value: Any, include_module: bool = True, serialize_function_as_dict: bool = True
) -> str | int | float | dict[str, Any] | list[Any] | bool | Any | None:
    """Return JSON serializable object.

    Args:
        value: object to serialize.
        include_module: include module in serialization.
        serialize_function_as_dict: serialize function as dict. False serializes as string.
    """
    from gdsfactory.path import Path

    if isinstance(value, pydantic.BaseModel):
        return clean_dict(value.model_dump(exclude_none=True))

    elif hasattr(value, "get_component_spec"):
        return value.get_component_spec()

    elif isinstance(value, bool):
        return value

    elif isinstance(value, Enum):
        return str(value)

    elif isinstance(value, np.integer | int):
        return int(value)

    elif isinstance(value, float | np.complexfloating | np.floating):
        if value == round(value):
            return int(value)
        return float(np.round(value, DEFAULT_SERIALIZATION_MAX_DIGITS))

    elif isinstance(value, complex | np.complexfloating):
        return complex_encoder(value)

    elif isinstance(value, np.ndarray):
        value = np.round(value, DEFAULT_SERIALIZATION_MAX_DIGITS)
        return orjson.loads(orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY))

    elif callable(value) and isinstance(value, functools.partial):
        return clean_value_partial(
            value=value,
            include_module=include_module,
            serialize_function_as_dict=serialize_function_as_dict,
        )
    elif hasattr(value, "to_dict"):
        return clean_dict(value.to_dict())

    elif callable(value) and isinstance(value, toolz.functoolz.Compose):
        return [clean_value_json(value.first)] + [
            clean_value_json(func) for func in value.funcs
        ]

    elif callable(value) and hasattr(value, "__name__"):
        if serialize_function_as_dict:
            return (
                {"function": value.__name__, "module": value.__module__}
                if include_module
                else {"function": value.__name__}
            )
        else:
            return value.__name__

    elif isinstance(value, Path):
        return value.hash_geometry()

    elif isinstance(value, pathlib.Path):
        return value.stem

    elif isinstance(value, dict):
        return clean_dict(value.copy())

    elif isinstance(value, list | tuple | set | KeysView):
        return tuple([clean_value_json(i) for i in value])

    elif attrs.has(type(value)):
        return attrs.asdict(value)

    else:
        try:
            value_json = orjson.dumps(
                value, option=orjson.OPT_SERIALIZE_NUMPY, default=clean_value_json
            )
            return orjson.loads(value_json)
        except TypeError as e:
            print(f"Error serializing {value!r}")
            raise e


def clean_value_partial(
    value: functools.partial[Any],
    include_module: bool = True,
    serialize_function_as_dict: bool = True,
) -> str | Any | dict[str, str | Any | dict[str, Any]]:
    sig = inspect.signature(value.func)
    args_as_kwargs = dict(zip(sig.parameters.keys(), value.args))
    args_as_kwargs |= value.keywords
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
    if not serialize_function_as_dict:
        return func.__name__
    return v


def clean_value_name(value: Any) -> str:
    """Returns a valid Python variable name representation of an object."""
    # Convert the value to a string and replace spaces with underscores
    cleaned = str(clean_value_json(value)).replace(" ", "_")

    # Remove invalid characters (only allow letters, numbers, and underscores)
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "", cleaned)

    # Ensure the name starts with a letter or underscore
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"var_{cleaned}"

    # Avoid reserved Python keywords
    if iskeyword(cleaned):
        cleaned = f"{cleaned}_var"

    return cleaned


def get_hash(value: Any) -> str:
    return hashlib.md5((clean_value_name(value)).encode()).hexdigest()[:8]


if __name__ == "__main__":
    from gdsfactory.components import straight

    s = clean_value_json(straight)
    print(s)

    # f = partial(gf.c.straight, length=3)
    # d = clean_value_json(f)
    # print(f"{d!r}")
    # f = partial(gf.c.straight, length=3)
    # c = f()
    # d = clean_value_json(c)
    # print(d, d)

    # xs = partial(
    #     gf.cross_section.strip,
    #     width=3,
    #     add_pins=gf.partial(gf.add_pins.add_pins_inside1nm, pin_length=0.1),
    # )
    # f = partial(gf.routing.add_fiber_array, cross_section=xs)
    # c = f()
    # c = gf.cross_section.strip(width=3)
    # d = clean_value_json(c)
    # print(get_hash(d))
