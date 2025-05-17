from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable
from functools import partial
from inspect import isfunction, signature
from typing import Any

from gdsfactory.component import Component
from gdsfactory.typings import ComponentFactory


def get_cells(
    modules: Any,
    ignore_non_decorated: bool = False,
    ignore_underscored: bool = True,
    ignore_partials: bool = False,
) -> dict[str, ComponentFactory]:
    """Returns PCells (component functions) from a module or list of modules.

    Args:
        modules: A module or an iterable of modules.
        ignore_non_decorated: only include functions that are decorated with gf.cell
        ignore_underscored: only include functions that do not start with '_'
        ignore_partials: only include functions, not partials
    """
    if not isinstance(modules, Iterable) or isinstance(modules, (str, bytes)):
        modules = [modules]

    cells: dict[str, ComponentFactory] = {}
    # Cached names for built-in/fast lookups
    _is_cell = is_cell

    for module in modules:
        # Faster than inspect.getmembers
        module_dict = getattr(module, "__dict__", None)
        if module_dict is not None:
            items = module_dict.items()
        else:
            items = ((name, getattr(module, name)) for name in dir(module))

        for name, member in items:
            if _is_cell(
                member,
                ignore_non_decorated=ignore_non_decorated,
                ignore_underscored=ignore_underscored,
                ignore_partials=ignore_partials,
                name=name,
            ):
                cells[name] = member

    return cells


def is_cell(
    func: Any,
    ignore_non_decorated: bool = False,
    ignore_underscored: bool = True,
    ignore_partials: bool = False,
    name: str = "",
) -> bool:
    # Fast unconditional check
    if not callable(func):
        return False

    # Handle functools.partial only if not skipping partials
    if not ignore_partials and isinstance(func, partial):
        return is_cell(
            func.func,
            ignore_non_decorated=ignore_non_decorated,
            ignore_underscored=ignore_underscored,
            ignore_partials=ignore_partials,
            name=name,
        )

    # Use attribute if sent (avoid double lookup)
    if not name:
        try:
            name = func.__name__
        except AttributeError:
            return False
    if ignore_underscored and name.startswith("_"):
        return False

    # Fast attribute check (most common)
    if getattr(func, "is_gf_cell", False):
        return True

    if not ignore_non_decorated:
        # signature() is expensive, only do if not already matched above
        try:
            r = func.__annotations__.get("return", None)
            if r is not None:
                # __annotations__ may have actual class ref or string
                if r is Component or (isinstance(r, str) and r.endswith("Component")):
                    return True
            # fallback if annotation isn't present/enough
            from inspect import signature

            r = signature(func).return_annotation
            return r is Component or (isinstance(r, str) and r.endswith("Component"))
        except Exception:
            return False
    return False


def get_cells_from_dict(
    cells: dict[str, Callable[..., Any]],
) -> dict[str, Callable[..., Component]]:
    """Returns PCells (component functions) from a dictionary.

    Args:
        cells: A dictionary of cells.

    Returns:
        A dictionary of valid component functions.
    """
    valid_cells: dict[str, Callable[..., Component]] = {}

    for name, member in cells.items():
        if not name.startswith("_") and (
            callable(member) and (isfunction(member) or isinstance(member, partial))
        ):
            with contextlib.suppress(ValueError):
                func = member.func if isinstance(member, partial) else member
                r = signature(func).return_annotation
                if r == Component or (isinstance(r, str) and r.endswith("Component")):
                    valid_cells[name] = member
    return valid_cells
