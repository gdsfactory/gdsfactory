"""Cross-section utility functions, factories, and registration."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial, wraps
from inspect import getmembers, isbuiltin, isfunction
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Any, ParamSpec, Protocol

import numpy as np
from kfactory import logger

from gdsfactory import typings
from gdsfactory.cross_section.base import (
    CrossSection,
    CrossSectionFactory,
    Section,
    Sections,
)

cross_sections: dict[str, CrossSectionFactory] = {}
_cross_section_default_names: dict[str, str] = {}

P = ParamSpec("P")


class CrossSectionCallable(Protocol[P]):
    __name__: str

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> CrossSection: ...


def xsection(
    func: CrossSectionCallable[P],
    xs_container: dict[str, CrossSectionFactory] = cross_sections,
    xs_default_mapping: dict[str, str] = _cross_section_default_names,
) -> CrossSectionCallable[P]:
    """Decorator to register a cross-section function.

    Ensures that the cross-section name matches the name of the function that generated it when created using default parameters

    .. code-block:: python

        @xsection
        def xs_sc(width=TECH.width_sc, radius=TECH.radius_sc):
            return gf.cross_section.cross_section(width=width, radius=radius)
    """
    default_xs = func()  # type: ignore[call-arg]
    xs_default_mapping[default_xs.name] = func.__name__

    @wraps(func)
    def newfunc(*args: P.args, **kwargs: P.kwargs) -> CrossSection:
        xs = func(*args, **kwargs)
        if xs.name in xs_default_mapping:
            xs._name = xs_default_mapping[xs.name]
        return xs

    xs_container[func.__name__] = newfunc
    return newfunc


def cross_section(
    width: float | typings.WidthFunction = 0.5,
    offset: float | typings.OffsetFunction = 0,
    layer: typings.LayerSpec = "WG",
    sections: Sections | None = None,
    port_names: typings.IOPorts = ("o1", "o2"),
    port_types: typings.IOPorts = ("optical", "optical"),
    bbox_layers: typings.LayerSpecs | None = None,
    bbox_offsets: typings.Floats | None = None,
    cladding_layers: typings.LayerSpecs | None = None,
    cladding_offsets: float | typings.Floats | None = None,
    cladding_simplify: float | typings.Floats | None = None,
    cladding_centers: float | typings.Floats | None = None,
    radius: float | None = 10.0,
    radius_min: float | None = 7.0,
    main_section_name: str = "_default",
) -> CrossSection:
    """Return CrossSection.

    Args:
        width: main Section width (um) or parameterized function from 0 to 1.
        offset: main Section center offset (um) or parameterized function from 0 to 1.
        layer: main section layer.
        sections: list of Sections(width, offset, layer, ports).
        port_names: for input and output ('o1', 'o2').
        port_types: for input and output: electrical, optical, vertical_te ...
        bbox_layers: list of layers bounding boxes to extrude.
        bbox_offsets: list of offset from bounding box edge.
        cladding_layers: list of layers to extrude.
        cladding_offsets: offset from main Section edge. Single float is
            broadcast to all cladding layers.
        cladding_simplify: Optional Tolerance value for the simplification algorithm. \
                All points that can be removed without changing the resulting. \
                polygon by more than the value listed here will be removed. \
                Single float is broadcast to all cladding layers.
        cladding_centers: center offset for each cladding layer. Defaults to 0. \
                Single float is broadcast to all cladding layers.
        radius: routing bend radius (um).
        radius_min: min acceptable bend radius.
        main_section_name: name of the main section. Defaults to _default

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.cross_section(width=0.5, offset=0, layer='WG')
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()

    .. code::


           ┌────────────────────────────────────────────────────────────┐
           │                                                            │
           │                                                            │
           │                   boox_layer                               │
           │                                                            │
           │         ┌──────────────────────────────────────┐           │
           │         │                            ▲         │bbox_offset│
           │         │                            │         ├──────────►│
           │         │           cladding_offset  │         │           │
           │         │                            │         │           │
           │         ├─────────────────────────▲──┴─────────┤           │
           │         │                         │            │           │
        ─ ─┤         │           core   width  │            │           ├─ ─ center
           │         │                         │            │           │
           │         ├─────────────────────────▼────────────┤           │
           │         │                                      │           │
           │         │                                      │           │
           │         │                                      │           │
           │         │                                      │           │
           │         └──────────────────────────────────────┘           │
           │                                                            │
           │                                                            │
           │                                                            │
           └────────────────────────────────────────────────────────────┘
    """
    section_list: list[Section] = list(sections or [])
    cladding_simplify_not_none: list[float | None] | None = None
    cladding_offsets_not_none: list[float] | None = None
    cladding_centers_not_none: list[float] | None = None
    if cladding_layers:

        def _broadcast(
            value: float | typings.Floats | None, default: float | None
        ) -> list[Any]:
            if isinstance(value, (int, float, np.number)):
                return [float(value)] * len(cladding_layers)
            if value is None or len(value) == 0:
                return [default] * len(cladding_layers)
            return list(value)

        cladding_simplify_not_none = _broadcast(cladding_simplify, None)
        cladding_offsets_not_none = _broadcast(cladding_offsets, 0)
        cladding_centers_not_none = _broadcast(cladding_centers, 0)

        if (
            len(
                {
                    len(x)
                    for x in (
                        cladding_layers,
                        cladding_offsets_not_none,
                        cladding_simplify_not_none,
                        cladding_centers_not_none,
                    )
                }
            )
            > 1
        ):
            raise ValueError(
                f"{len(cladding_layers)=}, "
                f"{len(cladding_offsets_not_none)=}, "
                f"{len(cladding_simplify_not_none)=}, "
                f"{len(cladding_centers_not_none)=} must have same length"
            )
    s = [
        Section(
            width=0 if callable(width) else width,
            width_function=width if callable(width) else None,
            offset=0 if callable(offset) else offset,
            offset_function=offset if callable(offset) else None,
            layer=layer,
            port_names=port_names,
            port_types=port_types,
            name=main_section_name,
        )
    ] + section_list

    if (
        cladding_layers
        and cladding_offsets_not_none
        and cladding_simplify_not_none
        and cladding_centers_not_none
    ):

        def _cladding_width_kwargs(offset: float) -> dict[str, Any]:
            if callable(width):
                return {"width_function": lambda t: width(t) + 2 * offset}
            return {"width": width + 2 * offset}

        s += [
            Section(
                **_cladding_width_kwargs(cladding_offset),
                layer=cladding_layer,
                simplify=cladding_simplify,
                offset=cladding_center,
                name=f"cladding_{i}",
            )
            for i, (
                cladding_layer,
                cladding_offset,
                cladding_simplify,
                cladding_center,
            ) in enumerate(
                zip(
                    cladding_layers,
                    cladding_offsets_not_none,
                    cladding_simplify_not_none,
                    cladding_centers_not_none,
                    strict=False,
                )
            )
        ]
    return CrossSection(
        sections=tuple(s),
        radius=radius,
        radius_min=radius_min,
        bbox_layers=bbox_layers,
        bbox_offsets=bbox_offsets,
    )


def is_cross_section(name: str, obj: Any, verbose: bool = False) -> bool:
    """Check if an object is a cross-section factory function.

    Args:
        name: Name of the object.
        obj: Object to check.
        verbose: Whether to print warnings for errors.

    Returns:
        True if the object is a cross-section factory function.
    """
    if name.startswith("_"):
        return False

    # Early prune: only consider functions, builtins or partials
    func: FunctionType | BuiltinFunctionType | None = None
    if isfunction(obj) or isbuiltin(obj):
        func = obj
    elif isinstance(obj, partial):
        # Check if the underlying function is a function or builtin
        if isfunction(obj.func) or isbuiltin(obj.func):
            func = obj.func
        else:
            return False
    else:
        return False

    # Ensure func is not None for type checker
    if func is None:
        return False

    # Check if function is registered in the cross_sections dictionary
    # This happens when decorated with @xsection
    if name in cross_sections and cross_sections[name] is obj:
        return True

    # Fallback: check return type annotation
    try:
        ann = getattr(func, "__annotations__", {})
        return_type = ann.get("return")

        if return_type is None:
            return False

        # Handle string annotations and forward references
        if isinstance(return_type, str):
            # Handle simple string matches
            if return_type in (
                "CrossSection",
                "gf.CrossSection",
                "gdsfactory.CrossSection",
            ):
                return True

            # For other string annotations, try to resolve them in the function's context
            try:
                # Try globals first
                func_globals = getattr(func, "__globals__", {})
                resolved_type = func_globals.get(return_type)

                # If not in globals, try closure variables
                if (
                    resolved_type is None
                    and hasattr(func, "__closure__")
                    and func.__closure__
                ):
                    # Get the names of closure variables
                    if hasattr(func, "__code__") and hasattr(
                        func.__code__, "co_freevars"
                    ):
                        freevars = func.__code__.co_freevars
                        closure_values = func.__closure__
                        if len(freevars) == len(closure_values):
                            closure_dict = dict(
                                zip(
                                    freevars,
                                    [cell.cell_contents for cell in closure_values],
                                    strict=False,
                                )
                            )
                            resolved_type = closure_dict.get(return_type)

                if resolved_type and isinstance(resolved_type, type):
                    return issubclass(resolved_type, CrossSection)

            except (TypeError, AttributeError, ValueError):
                pass  # Ignore type resolution errors

            return False

        # Direct type comparison
        if return_type is CrossSection:
            return True

        # Check if it's a subclass of CrossSection
        if isinstance(return_type, type):
            try:
                return issubclass(return_type, CrossSection)
            except TypeError:
                # Handle cases where return_type is not a class
                return False

    except Exception as e:
        if verbose:
            logger.warning(f"Error checking cross-section for {name}: {e}")

    return False


def get_cross_sections(
    modules: Sequence[ModuleType] | ModuleType, verbose: bool = False
) -> dict[str, CrossSectionFactory]:
    """Returns cross_sections from a module or list of modules.

    Args:
        modules: module or iterable of modules.
        verbose: prints in case any errors occur.
    """
    # Optimize module input normalization and preallocate xs
    if isinstance(modules, Sequence) and not isinstance(modules, str):
        modules_ = modules
    else:
        modules_ = [modules]

    xs: dict[str, CrossSectionFactory] = {
        name: obj
        for module in modules_
        for name, obj in getmembers(module)
        if is_cross_section(name, obj, verbose)
    }

    return xs


# cross_sections = get_cross_sections(sys.modules[__name__])
