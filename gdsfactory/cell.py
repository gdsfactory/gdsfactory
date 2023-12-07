"""Cell decorator for functions that return a Component."""
from __future__ import annotations

import functools
import hashlib
import inspect
from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

from pydantic import BaseModel, validate_call

from gdsfactory.component import Component, name_counters
from gdsfactory.config import CONF
from gdsfactory.name import clean_name, get_name_short
from gdsfactory.serialization import clean_dict, clean_value_name

CACHE: dict[str, Component] = {}
CACHE_IDS = set()

INFO_VERSION = 2

_F = TypeVar("_F", bound=Callable)


class CellReturnTypeError(ValueError):
    pass


def remove_from_cache(name: str | Component) -> None:
    """Removes Component name from CACHE and resets the name counter."""

    if not isinstance(name, str):
        name = name.name

    if name in CACHE:
        del CACHE[name]

    if name_counters[name] == 1:
        name_counters[name] = 0


def clear_cache() -> None:
    """Clears Component CACHE and reset the name counters."""

    CACHE.clear()
    CACHE_IDS.clear()
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
    max_name_length: int | None = None,
    include_module: bool = False,
    with_hash: bool = False,
    ports_off_grid: str | None = None,
    ports_not_manhattan: str | None = None,
    flatten: bool = False,
    naming_style: str = "default",
    default_decorator: Callable[[Component], Component] | None = None,
    add_settings: bool = True,
    validate: bool = False,
    get_child_name: bool = False,
) -> Callable[[_F], _F]:
    """Parametrized Decorator for Component functions.

    Args:
        func: function to decorate.
        autoname: True renames Component based on args and kwargs. True by default.
        max_name_length: truncates name beyond some characters with a hash. Defaults to CONF.max_name_length.
        include_module: True adds module name to the cell name.
        with_hash: True adds a hash to the cell name.
        ports_off_grid: "warn", "error" or "ignore". Checks if ports are on grid. Defaults to CONF.ports_off_grid.
        ports_not_manhattan: "warn", "error" or "ignore". Checks if ports are manhattan. Defaults to CONF.ports_non_manhattan.
        flatten: False by default. True flattens component hierarchy.
        naming_style: "default" or "updk". "default" is the default naming style.
        default_decorator: default decorator to apply to the component. None by default.
        add_settings: True by default. Adds settings to the component.
        validate: validate the function call. Does not work with annotations that have None | Callable.
        get_child_name: Use child name as component name prefix.

    Implements a cache so that if a component has already been build it returns the component from the cache directly.
    This avoids creating two exact Components that have the same name.
    Can autoname components based on the function name and arguments.

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
        nonlocal ports_not_manhattan, ports_off_grid, max_name_length
        from gdsfactory.pdk import get_active_pdk

        active_pdk = get_active_pdk()

        info = kwargs.pop("info", {})  # TODO: remove info
        name = _name = kwargs.pop("name", None)  # TODO: remove name
        prefix = kwargs.pop("prefix", func.__name__)  # TODO: remove prefix
        sig = inspect.signature(func)
        args_as_kwargs = dict(zip(sig.parameters.keys(), args))
        args_as_kwargs.update(kwargs)

        if max_name_length is None:
            max_name_length = CONF.max_name_length
        if ports_off_grid is None:
            ports_off_grid = CONF.ports_off_grid
        if ports_not_manhattan is None:
            ports_not_manhattan = CONF.ports_not_manhattan

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
        # if no decorator is specified, but there is one specified for the active PDK, use the PDK's default decorator
        if decorator is None and active_pdk.default_decorator is not None:
            decorator = active_pdk.default_decorator

        if name in CACHE:
            # print(f"CACHE LOAD {name} {func.__name__}({named_args_string})")
            return CACHE[name]

        # print(f"BUILD {name} {func.__name__}({named_args_string})")
        if not callable(func):
            raise ValueError(
                f"{func!r} is not callable! @cell decorator is only for functions"
            )

        if validate:
            component = validate_call(func)(*args, **kwargs)

        else:
            component = func(*args, **kwargs)

        if ports_off_grid in ("warn", "error"):
            component.assert_ports_on_grid(error_type=ports_off_grid)
        if ports_not_manhattan in ("warn", "error"):
            component.assert_ports_manhattan(error_type=ports_not_manhattan)
        if flatten:
            component = component.flatten()

        # if the component is already in the cache, but under a different alias,
        # make sure we use a copy, so we don't run into mutability errors
        if id(component) in CACHE_IDS:
            component = component.copy()

        if not isinstance(component, Component):
            raise CellReturnTypeError(
                f"function {func.__name__!r} return type = {type(component)}",
                "make sure that functions with @cell decorator return a Component",
            )

        metadata_child = None
        if get_child_name and _name is None:
            if component.child is None:
                raise ValueError(
                    f"{name}: get_child_name was defined, but component has no child! Be sure to assign the component a child attribute."
                )
            metadata_child = dict(component.child.settings)
            component_name = f"{metadata_child.get('name')}_{name}"
            component_name = get_name_short(
                component_name, max_name_length=max_name_length
            )
            # if cache and component_name in CACHE:
            # return CACHE[component_name]
        else:
            component_name = name

        if autoname:
            component.rename(component_name, max_name_length=max_name_length)
        if get_child_name:
            CACHE[name] = component

        info = info or {}
        component.info.update(**info)
        if add_settings:
            component.settings = Settings(
                name=component_name,
                function_name=func.__name__,
                module=func.__module__,
                changed=clean_dict(changed),
                default=clean_dict(default),
                full=clean_dict(full),
                info=component.info,
            )
            component.__doc__ = func.__doc__

        if decorator:
            if not callable(decorator):
                raise ValueError(f"decorator = {type(decorator)} needs to be callable")
            component_new = decorator(component)
            component = component_new or component

        component.lock()
        CACHE_IDS.add(id(component))
        return component

    return (
        wrapper
        if func is not None
        else partial(
            cell,
            autoname=autoname,
            max_name_length=max_name_length,
            include_module=include_module,
            with_hash=with_hash,
            ports_off_grid=ports_off_grid,
            ports_not_manhattan=ports_not_manhattan,
            flatten=flatten,
            naming_style=naming_style,
            default_decorator=default_decorator,
            add_settings=add_settings,
            validate=validate,
            get_child_name=get_child_name,
        )
    )


cell_without_validator = cell
cell_with_module = partial(cell, include_module=True)
cell_import_gds = partial(cell, autoname=False, add_settings=False)
cell_with_child = partial(cell, get_child_name=True)


@cell_with_child
def container(component, function, **kwargs) -> gf.Component:
    """Returns new component with a component reference.

    Args:
        component: to add to container.
        function: function to apply to component.
        kwargs: keyword arguments to pass to function.

    """
    import gdsfactory as gf

    component = gf.get_component(component)
    c = Component()
    cref = c << component
    function(c, **kwargs)
    c.ports = cref.ports
    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.straight(info={"simulation": "eme"}, name="hi")
    print(c.name)
    # print(c.info["simulation"])
    c.show()
