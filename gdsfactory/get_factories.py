from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from inspect import getmembers, isfunction, signature
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
    modules = modules if isinstance(modules, Iterable) else [modules]
    cells: dict[str, ComponentFactory] = {}
    for module in modules:
        cells.update(
            {
                name: member
                for name, member in getmembers(module)
                if is_cell(
                    member,
                    ignore_non_decorated=ignore_non_decorated,
                    ignore_underscored=ignore_underscored,
                    ignore_partials=ignore_partials,
                    name=name,
                )
            }
        )
    return cells


def is_cell(
    func: Any,
    ignore_non_decorated: bool = False,
    ignore_underscored: bool = True,
    ignore_partials: bool = False,
    name: str = "",
) -> bool:
    try:
        if not callable(func):
            return False
        if not ignore_partials and isinstance(func, partial):
            return is_cell(
                func.func,
                ignore_non_decorated=ignore_non_decorated,
                ignore_underscored=ignore_underscored,
                ignore_partials=ignore_partials,
                name=name,
            )
        if not name:
            name = func.__name__
        if ignore_underscored and name.startswith("_"):
            return False
        if getattr(func, "is_gf_cell", False):
            return True
        if not ignore_non_decorated:
            r = signature(func).return_annotation
            return r == Component or (isinstance(r, str) and r.endswith("Component"))
    except Exception:
        pass
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
    _Component = Component  # local lookup is slightly faster

    for name, member in cells.items():
        # Fastpath: skip names starting with '_' early
        if name and name[0] == "_":
            continue

        # Fastpath: skip non-callable objects early
        if not callable(member):
            continue

        # isfunction/member check
        is_func = isfunction(member)
        is_partial = isinstance(member, partial)
        if not (is_func or is_partial):
            continue

        # Get underlying function for partial, else the member itself
        func = member.func if is_partial else member

        # Try-block only around the annotation lookup
        try:
            r = _get_signature_return_annotation(func)
        except ValueError:
            continue

        # Fastpath: check object identity (faster than isinstance)
        if r is _Component or (isinstance(r, str) and r.endswith("Component")):
            valid_cells[name] = member

    return valid_cells


def _get_signature_return_annotation(func):
    """Helper to cache and retrieve the return annotation for a function."""
    try:
        return _signature_cache[func]
    except KeyError:
        # Save in cache; inspect.signature is very expensive
        r = signature(func).return_annotation
        _signature_cache[func] = r
        return r


_signature_cache = {}
