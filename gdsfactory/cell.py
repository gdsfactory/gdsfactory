"""Cell decorator for functions that return a Component."""
from __future__ import annotations

import functools
import hashlib
import inspect
from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

from pydantic import BaseModel

from gdsfactory.component import Component, name_counters
from gdsfactory.config import CONF
from gdsfactory.name import clean_name, get_name_short
from gdsfactory.serialization import clean_dict, clean_value_name

CACHE: dict[str, Component] = {}

INFO_VERSION = 2

_F = TypeVar("_F", bound=Callable)


class CellReturnTypeError(ValueError):
    pass


def clear_cache() -> None:
    """Clears Component CACHE."""
    global CACHE

    CACHE = {}
    name_counters.clear()


def print_cache() -> None:
    for k in CACHE:
        print(k)


class Settings(BaseModel):
    name: str
    function_name: str | None = None
    module: str | None = None

    info: dict[str, Any] = {}  # derived properties (length, resistance)
    info_version: int = INFO_VERSION

    full: dict[str, Any] = {}
    changed: dict[str, Any] = {}
    default: dict[str, Any] = {}

    child: dict[str, Any] | None = None


def cell(
    func: _F | None = None,
    /,
    *,
    autoname: bool = True,
    max_name_length: int = CONF.max_name_length,
    include_module: bool = False,
    with_hash: bool = False,
    ports_off_grid: str = CONF.ports_off_grid,
    ports_not_manhattan: str = CONF.ports_not_manhattan,
    flatten: bool = False,
    naming_style: str = "default",
    default_decorator: Callable[[Component], Component] | None = None,
) -> Callable[[_F], _F]:
    """Parametrized Decorator for Component functions.

    decorator_kwargs are the default settings that the decorator will use, they can be overridden by the decorated function's kwargs.

    Args:
        autoname: True renames Component based on args and kwargs. True by default.
        max_name_length: truncates name beyond some characters with a hash.
        include_module: True adds module name to the cell name.
        with_hash: True adds a hash to the cell name.
        ports_off_grid: "warn", "error" or "ignore". Checks if ports are on grid.
        ports_not_manhattan: "warn", "error" or "ignore". Checks if ports are manhattan.
        flatten: False by default. True flattens component hierarchy.
        naming_style: "default" or "updk". "default" is the default naming style.
        default_decorator: default decorator to apply to the component. None by default.

    kwargs:
        cache: avoids creating duplicated Components.
        name: names Components uniquely name based on parameters.

    Implements a cache so that if a component has already been build it returns the component from the cache directly.
    This avoids creating two exact Components that have the same name.

    A decorator is a function that runs over a function, so when you do.

    .. code::

        import gdsfactory as gf

        @gf.cell
        def mzi_with_bend():
            c = gf.Component()
            mzi = c << gf.components.mzi()
            bend = c << gf.components.bend_euler()
            return c

    it’s equivalent to

    .. code::

        def mzi_with_bend():
            c = gf.Component()
            mzi = c << gf.components.mzi()
            bend = c << gf.components.bend_euler(radius=radius)
            return c

        mzi_with_bend_decorated = gf.cell(mzi_with_bend)

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Component:
        name = kwargs.pop("name", None)
        cache = kwargs.pop("cache", True)
        prefix = kwargs.pop("prefix", func.__name__)
        sig = inspect.signature(func)
        args_as_kwargs = dict(zip(sig.parameters.keys(), args))
        args_as_kwargs.update(kwargs)

        default = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.default != inspect._empty
        }

        changed = args_as_kwargs
        full = default.copy()
        full.update(**args_as_kwargs)
        default2 = default.copy()
        changed2 = changed.copy()

        # list of default args as strings
        default_args_list = [
            f"{key}={clean_value_name(default2[key])}" for key in sorted(default.keys())
        ]
        # list of explicitly passed args as strings
        passed_args_list = [
            f"{key}={clean_value_name(changed2[key])}" for key in sorted(changed.keys())
        ]

        if naming_style == "updk":
            full_args_list = [
                f"{key}={clean_value_name(full[key])}" for key in sorted(full.keys())
            ]
            named_args_string = ",".join(full_args_list)
            name = f"{prefix}:{named_args_string}" if named_args_string else prefix
            name = clean_name(name, allowed_characters=[":", ".", "="])

        elif naming_style == "default":
            changed_arg_set = set(passed_args_list).difference(default_args_list)
            changed_arg_list = sorted(changed_arg_set)
            named_args_string = "_".join(changed_arg_list)

            if include_module:
                named_args_string += f"_{func.__module__}"
            if changed_arg_list:
                named_args_string = (
                    hashlib.md5(named_args_string.encode()).hexdigest()[:8]
                    if with_hash
                    or len(named_args_string) > 28
                    or "'" in named_args_string
                    or "{" in named_args_string
                    else named_args_string
                )

            name_signature = (
                clean_name(f"{prefix}_{named_args_string}")
                if named_args_string
                else clean_value_name(prefix)
            )
            # filter the changed dictionary to only keep entries which have truly changed
            changed_arg_names = [carg.split("=")[0] for carg in changed_arg_list]
            changed = {k: changed[k] for k in changed_arg_names}
            name = name or name_signature

        else:
            raise ValueError('naming_style must be "default" or "updk"')

        name = get_name_short(name, max_name_length=max_name_length)
        decorator = kwargs.pop("decorator", default_decorator)

        if (
            "args" not in sig.parameters
            and "kwargs" not in sig.parameters
            and "settings" not in sig.parameters
        ):
            for key in kwargs:
                if key not in sig.parameters.keys():
                    raise TypeError(
                        f"{func.__name__!r}() got invalid argument {key!r}\n"
                        f"valid arguments are {list(sig.parameters.keys())}"
                    )

        if cache and name in CACHE:
            # print(f"CACHE LOAD {name} {func.__name__}({named_args_string})")
            return CACHE[name]

        # print(f"BUILD {name} {func.__name__}({named_args_string})")
        if not callable(func):
            raise ValueError(
                f"{func!r} is not callable! @cell decorator is only for functions"
            )

        component = func(*args, **kwargs)

        if ports_off_grid in ("warn", "error"):
            component.assert_ports_on_grid(error_type=ports_off_grid)
        if ports_not_manhattan in ("warn", "error"):
            component.assert_ports_manhattan(error_type=ports_not_manhattan)
        if flatten:
            component = component.flatten()

        # if the component is already in the cache, but under a different alias,
        # make sure we use a copy, so we don't run into mutability errors
        if id(component) in [id(v) for v in CACHE.values()]:
            component = component.copy()

        metadata_child = (
            dict(component.child.settings) if hasattr(component, "child") else None
        )

        if not isinstance(component, Component):
            raise CellReturnTypeError(
                f"function {func.__name__!r} return type = {type(component)}",
                "make sure that functions with @cell decorator return a Component",
            )

        if metadata_child and component._get_child_name:
            component_name = f"{metadata_child.get('name')}_{name}"
            component_name = get_name_short(
                component_name, max_name_length=max_name_length
            )
        else:
            component_name = name

        if autoname and not hasattr(component, "imported_gds"):
            component.name = component_name

        if not hasattr(component, "imported_gds"):
            component.settings = Settings(
                name=component_name,
                function_name=func.__name__,
                module=func.__module__,
                changed=clean_dict(changed),
                default=clean_dict(default),
                full=clean_dict(full),
                info=component.info,
                child=metadata_child,
            )
            component.__doc__ = func.__doc__

        if decorator:
            if not callable(decorator):
                raise ValueError(f"decorator = {type(decorator)} needs to be callable")
            component_new = decorator(component)
            component = component_new or component

        component.lock()
        CACHE[name] = component
        return component

    return wrapper if func is not None else cell


cell_without_validator = cell
cell_with_module = partial(cell, include_module=True)


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.straight()
    c.show()
