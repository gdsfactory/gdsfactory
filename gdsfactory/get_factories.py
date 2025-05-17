from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable
from functools import lru_cache, partial
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
    # Ensure modules is a list (avoid checking in each loop iteration)
    if not isinstance(modules, Iterable) or isinstance(modules, (str, bytes)):
        modules = [modules]

    cells: dict[str, ComponentFactory] = {}
    # Cached names for built-in/fast lookups
    _is_cell = is_cell

    for module in modules:
        # Prefer module.__dict__ for speed, else fallback to getattr walk
        module_dict = getattr(module, "__dict__", None)
        if module_dict is not None:
            items = module_dict.items()
        else:
            # generator for (name, value) from dir/module
            items = ((name, getattr(module, name)) for name in dir(module))

        # Incrementally add to cells (so multiple modules don't overwrite each other entirely)
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

    # Fast partial check: skip recursion if needed
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
        name = getattr(func, "__name__", None)
        if name is None:
            return False

    if ignore_underscored and name.startswith("_"):
        return False

    # Fast attribute check for 'is_gf_cell'
    is_Gf_cell = getattr(func, "is_gf_cell", False)
    if is_Gf_cell:
        return True

    # Only proceed to expensive checks if needed
    if not ignore_non_decorated:
        ann = getattr(func, "__annotations__", None)
        if ann:
            ann_ret = ann.get("return", None)
            if ann_ret is Component or (
                isinstance(ann_ret, str) and ann_ret.endswith("Component")
            ):
                return True
        # Use cached signature resolution
        try:
            r = _get_signature(func).return_annotation
            if r is Component or (isinstance(r, str) and r.endswith("Component")):
                return True
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


@lru_cache(maxsize=2048)
def _cached_signature(func_id):
    """Cache inspect.signature by function id."""
    # 'func_id' is a tuple: (id(func), type(func))
    return signature(_cached_signature._func_map[func_id])


def _get_signature(func):
    fid = (id(func), type(func))
    _cached_signature._func_map[fid] = func
    return _cached_signature(fid)
