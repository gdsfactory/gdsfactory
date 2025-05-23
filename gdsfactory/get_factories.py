from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable
from functools import lru_cache, partial
from inspect import Signature, getmembers, isfunction, ismethod, signature
from typing import Any

import kfactory as kf

from gdsfactory.component import Component, ComponentAllAngle
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
        ignore_non_decorated: only include functions that are decorated with gf.cell.
        ignore_underscored: only include functions that do not start with '_'.
        ignore_partials: only include functions, not partials.
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
    """Checks if a function is a GDSFactory cell.

    Args:
        func: The function to check.
        ignore_non_decorated: only include functions that are decorated with gf.cell.
        ignore_underscored: only include functions that do not start with '_'.
        ignore_partials: only include functions, not partials.
        name: The name of the function.
    """
    try:
        # Fast fail for non-callable
        if not callable(func):
            return False

        # Handle functools.partial recursively
        if not ignore_partials and isinstance(func, partial):
            return is_cell(
                func.func,
                ignore_non_decorated=ignore_non_decorated,
                ignore_underscored=ignore_underscored,
                ignore_partials=ignore_partials,
                name=name,
            )

        # Delay __name__ resolution until needed
        if not name:
            # __name__ is only on functions/methods/partial
            try:
                name = func.__name__
            except AttributeError:
                # fallback, won't match underscored, continue
                name = ""

        if ignore_underscored and name.startswith("_"):
            return False

        # Fast attribute check
        is_cell_attr = getattr(func, "is_gf_cell", None)

        if func in kf.kcl.virtual_factories.values():
            return True

        if is_cell_attr:
            return True

        if not ignore_non_decorated:
            # Use fast path for functions/methods with __annotations__
            return_annotation = None
            if isfunction(func) or ismethod(func):
                return_annotation = func.__annotations__.get("return", None)
            else:
                # Not a plain python function, fallback to full signature
                try:
                    return_annotation = _get_signature(func).return_annotation
                except Exception:
                    pass  # leave as None

            # Compare annotation to Component
            if return_annotation is not None:
                # Support both direct and string annotations
                if return_annotation == Component:
                    return True
                if isinstance(return_annotation, str) and return_annotation.endswith(
                    "Component"
                ):
                    return True
                if return_annotation == ComponentAllAngle:
                    return True
                if isinstance(return_annotation, str) and return_annotation.endswith(
                    "ComponentAllAngle"
                ):
                    return True
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
def _get_signature(obj: Any) -> Signature:
    """Cache signatures to avoid repeated work."""
    return signature(obj)
