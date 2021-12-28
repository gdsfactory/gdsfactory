"""Define names, clean values for names.
"""
import functools
import hashlib
import inspect
from typing import Any, Iterable

import numpy as np
import pydantic
import toolz
from phidl import Device, Port
from phidl.device_layout import Path as PathPhidl

from gdsfactory.hash_points import hash_points
from gdsfactory.snap import snap_to_grid

MAX_NAME_LENGTH = 32


@pydantic.validate_arguments
def get_name_short(name: str, max_name_length=MAX_NAME_LENGTH) -> str:
    """Returns a short name."""
    if len(name) > max_name_length:
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        name = f"{name[:(max_name_length - 9)]}_{name_hash}"
    return name


def join_first_letters(name: str) -> str:
    """Join the first letter of a name separated with underscores.

    taper_length -> TL
    """
    return "".join([x[0] for x in name.split("_") if x])


# replace function_name prefix for some components
component_type_to_name = dict(phidl="phidl")


def get_component_name(component_type: str, *args, **kwargs) -> str:
    """Returns concatenated kwargs Key_Value."""
    name = component_type
    name += "_".join([clean_value(a) for a in args])
    for k, v in component_type_to_name.items():
        name = name.replace(k, v)
    if kwargs:
        name += "_" + dict2name(**kwargs)
    return name


def dict2hash(**kwargs) -> str:
    ignore_from_name = kwargs.pop("ignore_from_name", [])
    h = hashlib.sha256()
    for key in sorted(kwargs):
        if key not in ignore_from_name:
            value = kwargs[key]
            value = clean_value(value)
            h.update(f"{key}{value}".encode())
    return h.hexdigest()


def dict2name(prefix: str = "", **kwargs) -> str:
    """Returns name from a dict."""
    ignore_from_name = kwargs.pop("ignore_from_name", [])
    kv = []
    kwargs = kwargs.copy()
    kwargs.pop("layer_to_inclusion", "")

    for key in sorted(kwargs):
        if key not in ignore_from_name and isinstance(key, str):
            value = kwargs[key]
            # key = join_first_letters(key).upper()
            if value is not None:
                kv += [f"{key}{clean_value(value)}"]
    label = prefix + "_".join(kv)
    return clean_name(label)


def assert_first_letters_are_different(**kwargs):
    """Assert that the first letters for each key are different.

    Avoids name collisions of different args that start with the same first letter.
    """
    first_letters = [join_first_letters(k) for k in kwargs.keys()]
    if not len(set(first_letters)) == len(first_letters):
        raise ValueError(
            f"Possible name collision! {kwargs.keys()} repeats first letters {first_letters}",
            "you can separate your arguments with underscores",
            " (delta_length -> DL, delta_width -> DW",
        )


def print_first_letters_warning(**kwargs):
    """Prints kwargs that have same cell."""
    first_letters = [join_first_letters(k) for k in kwargs.keys()]
    if not len(set(first_letters)) == len(first_letters):
        print(
            f"Possible name collision! {kwargs.keys()} "
            f"repeats first letters {first_letters}"
            "you can separate your arguments with underscores"
            " (delta_length -> DL, delta_width -> DW"
        )


def clean_name(name: str) -> str:
    """Return a string with correct characters for a cell name.

    [a-zA-Z0-9]

    FIXME: only a few characters are currently replaced.
        This function has been updated only on case-by-case basis
    """
    replace_map = {
        " ": "_",
        "!": "",
        "?": "",
        "#": "_",
        "%": "_",
        "(": "",
        ")": "",
        "*": "_",
        ",": "_",
        "-": "m",
        ".": "p",
        "/": "_",
        ":": "_",
        "=": "",
        "@": "_",
        "[": "",
        "]": "",
        "$": "",
    }
    for k, v in list(replace_map.items()):
        name = name.replace(k, v)
    return name


def clean_value(value: Any) -> str:
    """returns more readable value (integer)
    if number is < 1:
        returns number units in nm (integer)

    units are in um by default. Therefore when we multiply by 1e3 we get nm.
    """

    if isinstance(value, int):
        value = str(value)
    elif isinstance(value, (float, np.float64)):
        if 1 > value > 1e-3:
            value = f"{int(value*1e3)}n"
        elif int(value) == value:
            value = str(int(value))
        elif 1e-6 < value < 1e-3:
            value = f"{snap_to_grid(value*1e6)}u"
        elif 1e-9 < value < 1e-6:
            value = f"{snap_to_grid(value*1e9)}n"
        elif 1e-12 < value < 1e-9:
            value = f"{snap_to_grid(value*1e12)}p"
        else:  # Any unit < 1pm will disappear
            value = str(snap_to_grid(value)).replace(".", "p")
    elif isinstance(value, Device):
        value = clean_name(value.name)
    elif isinstance(value, str):
        value = value.strip()
    elif isinstance(value, dict):
        value = dict2name(**value)
        # value = [f"{k}={v!r}" for k, v in value.items()]
    elif isinstance(value, Port):
        value = f"{value.name}_{value.width}_{value.x}_{value.y}"
    elif isinstance(value, PathPhidl):
        value = f"path_{hash_points(value.points)}"
    elif (
        isinstance(value, object)
        and hasattr(value, "name")
        and isinstance(value.name, str)
    ):
        value = clean_name(value.name)
    elif callable(value) and isinstance(value, functools.partial):
        sig = inspect.signature(value.func)
        args_as_kwargs = dict(zip(sig.parameters.keys(), value.args))
        args_as_kwargs.update(**value.keywords)
        value = value.func.__name__ + dict2name(**args_as_kwargs)
    elif callable(value) and isinstance(value, toolz.functoolz.Compose):
        value = "_".join(
            [clean_value(v) for v in value.funcs] + [clean_value(value.first)]
        )
    elif callable(value) and hasattr(value, "__name__"):
        value = value.__name__
    elif hasattr(value, "get_name"):
        value = value.get_name()
    elif isinstance(value, Iterable):
        value = "_".join(clean_value(v) for v in value)

    return str(value)


def test_clean_value() -> None:
    assert clean_value(0.5) == "500n"
    assert clean_value(5) == "5"
    assert clean_value(5.0) == "5"
    assert clean_value(11.001) == "11p001"


def test_clean_name() -> None:
    assert clean_name("wg(:_=_2852") == "wg___2852"


if __name__ == "__main__":
    # test_cell()
    test_clean_value()
    import gdsfactory as gf

    # print(clean_value(gf.components.straight))
    # c = gf.components.straight(polarization="TMeraer")
    # print(c.settings["polarization"])
    # print(clean_value(11.001))
    # layers_cladding = (gf.LAYER.WGCLAD, gf.LAYER.NO_TILE_SI)
    # layers_cladding = (gf.LAYER.WGCLAD,)
    c = gf.components.straight(length=10)
    c = gf.components.straight(length=10)

    # print(c.name)
    # print(c)
    # c.show()

    # print(clean_name("Waveguidenol1_(:_=_2852"))
    # print(clean_value(1.2))
    # print(clean_value(0.2))
    # print(clean_value([1, [2.4324324, 3]]))
    # print(clean_value([1, 2.4324324, 3]))
    # print(clean_value((0.001, 24)))
    # print(clean_value({"a": 1, "b": 2}))
