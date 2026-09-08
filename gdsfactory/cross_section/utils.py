"""Cross-section utility functions, factories, and registration."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from functools import partial, wraps
from inspect import getmembers, isbuiltin, isfunction
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Any, ParamSpec, Protocol

import kfactory as kf
import numpy as np
from kfactory import logger

from gdsfactory import typings
from gdsfactory.cross_section.base import (
    CrossSection,
    CrossSectionFactory,
    LegacyCrossSection,
    Section,
)
from gdsfactory.cross_section.kfactory import (
    KFactorySectionSpec,
    _canonical_name,
    kfactory_cross_section,
)

cross_sections: dict[str, CrossSectionFactory] = {}
_cross_section_default_names: dict[str, str] = {}

P = ParamSpec("P")


class CrossSectionWarning(DeprecationWarning):
    """Warning emitted while adapting legacy cross-section metadata."""


class CrossSectionCallable(Protocol[P]):
    __name__: str

    def __call__(
        self, *args: P.args, **kwargs: P.kwargs
    ) -> CrossSection | LegacyCrossSection: ...


# Keep the old protocol name available to downstream imports during the
# migration.  Runtime factories now return native kfactory profiles.
LegacyCrossSectionCallable = CrossSectionCallable


def _warn(message: str, *, stacklevel: int = 3) -> None:
    warnings.warn(message, CrossSectionWarning, stacklevel=stacklevel)


def _nominal_value(
    value: float | Callable[..., Any], parameter_name: str, *, warn: bool
) -> float:
    if not callable(value):
        return float(value)

    if warn:
        _warn(
            f"{parameter_name} callable is not supported by native kfactory "
            "cross-sections; evaluating it at t=0.5 for the temporary adapter."
        )
    result = value(0.5)
    result_array = np.asarray(result)
    if result_array.size != 1:
        raise ValueError(
            f"{parameter_name} callable must return a scalar for the native "
            f"cross-section adapter, got shape {result_array.shape}."
        )
    return float(result_array.reshape(-1)[0])


def _section_to_kfactory_spec(section: Section, *, warn: bool) -> KFactorySectionSpec:
    width = _nominal_value(
        section.width_function or section.width, "section.width", warn=warn
    )
    offset = _nominal_value(
        section.offset_function or section.offset, "section.offset", warn=warn
    )
    if warn:
        _warn(
            "Legacy Section metadata (port names/types, simplify, hidden, "
            "insets, and transition flags) is dropped by the temporary native "
            "cross-section adapter."
        )
    return section.layer, offset - width / 2, offset + width / 2


def _rename_native_cross_section(
    cross_section: CrossSection, name: str | None
) -> CrossSection:
    if name is None or cross_section.name == name:
        return cross_section

    if isinstance(cross_section, kf.DCrossSection):
        sections: list[tuple[Any, float] | tuple[Any, float, float]] = []
        for layer, layer_sections in cross_section.sections.items():
            for section_min, section_max in layer_sections:
                if section_min is None:
                    sections.append((layer, section_max))
                else:
                    sections.append((layer, section_min, section_max))
        bbox_layers = list(cross_section.bbox_sections)
        try:
            return kf.DCrossSection(
                kcl=cross_section.kcl,
                width=cross_section.width,
                layer=cross_section.layer,
                sections=sections,
                bbox_layers=bbox_layers,
                bbox_offsets=[
                    cross_section.bbox_sections[layer] for layer in bbox_layers
                ],
                radius=cross_section.radius,
                radius_min=cross_section.radius_min,
                name=name,
            )
        except kf.exceptions.CrossSectionNamingConflictError:
            _warn(
                f"Could not rename native cross-section {_canonical_name(cross_section)!r} "
                f"to {name!r} because the name is already in use; keeping the "
                f"canonical name {cross_section.name!r}."
            )
            return cross_section
    if isinstance(cross_section, kf.DAsymmetricCrossSection):
        try:
            return kf.DAsymmetricCrossSection(
                kcl=cross_section.kcl,
                section_min=cross_section.section_min,
                section_max=cross_section.section_max,
                layer=cross_section.layer,
                sections=cross_section.sections,
                bbox_sections=cross_section.bbox_sections,
                radius=cross_section.radius,
                radius_min=cross_section.radius_min,
                name=name,
            )
        except kf.exceptions.CrossSectionNamingConflictError:
            _warn(
                f"Could not rename native cross-section {_canonical_name(cross_section)!r} "
                f"to {name!r} because the name is already in use; keeping the "
                f"canonical name {cross_section.name!r}."
            )
            return cross_section
    raise TypeError(f"Unsupported native cross-section type: {type(cross_section)}")


def _to_native_cross_section(
    cross_section: CrossSection | LegacyCrossSection,
    *,
    name: str | None = None,
    warn: bool = True,
) -> CrossSection:
    """Adapt a legacy profile to a native kfactory cross-section.

    The adapter intentionally preserves only geometry, radius, and bounding
    box information.  Port and extrusion metadata will move to the extrusion
    APIs in a later migration step.
    """
    if isinstance(cross_section, kf.DCrossSection | kf.DAsymmetricCrossSection):
        return _rename_native_cross_section(cross_section, name)

    if not isinstance(cross_section, LegacyCrossSection):
        raise TypeError(
            "Cross-section factories must return a native CrossSection or "
            f"LegacyCrossSection, got {type(cross_section)}."
        )

    if warn:
        _warn(
            "LegacyCrossSection is being adapted to a native kfactory "
            "cross-section; extrusion metadata is dropped temporarily."
        )

    if not cross_section.sections:
        raise ValueError("LegacyCrossSection must contain at least one section.")

    main = cross_section.sections[0]
    width = _nominal_value(
        main.width_function or main.width, "section.width", warn=warn
    )
    offset = _nominal_value(
        main.offset_function or main.offset, "section.offset", warn=warn
    )
    sections = tuple(
        _section_to_kfactory_spec(section, warn=warn)
        for section in cross_section.sections[1:]
    )
    return kfactory_cross_section(
        width=width,
        offset=offset,
        layer=main.layer,
        sections=sections,
        bbox_layers=cross_section.bbox_layers,
        bbox_offsets=cross_section.bbox_offsets,
        radius=cross_section.radius,
        radius_min=cross_section.radius_min,
        name=name,
    )


def _call_cross_section_factory(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    warn: bool,
) -> CrossSection:
    result = func(*args, **kwargs)
    return _to_native_cross_section(result, warn=warn)


def _call_cross_section_factory_without_warnings(
    func: Callable[..., Any],
) -> CrossSection:
    """Resolve a factory default without warning during lazy registration."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CrossSectionWarning)
        result = func()
    return _to_native_cross_section(result, warn=False)


def xsection[**P](
    func: CrossSectionCallable[P],
    xs_container: dict[str, CrossSectionFactory] = cross_sections,
    xs_default_mapping: dict[str, str] = _cross_section_default_names,
) -> Callable[P, CrossSection]:
    """Decorator to register a cross-section function.

    Ensures that the cross-section name matches the name of the function that generated it when created using default parameters

        @xsection
        def xs_sc(width=TECH.width_sc, radius=TECH.radius_sc):
            return gf.cross_section.cross_section(width=width, radius=radius)
    """
    default_xs_name: str | None = None

    @wraps(func)
    def newfunc(*args: P.args, **kwargs: P.kwargs) -> CrossSection:
        nonlocal default_xs_name
        if default_xs_name is None:
            default_xs = _call_cross_section_factory_without_warnings(func)
            default_xs_name = default_xs.name
            xs_default_mapping[default_xs_name] = func.__name__
        xs = _call_cross_section_factory(func, args, kwargs, warn=True)
        if xs.name == default_xs_name and not xs.base.is_named:
            xs = _rename_native_cross_section(xs, func.__name__)
        return xs

    xs_container[func.__name__] = newfunc
    return newfunc


def cross_section(
    width: float | typings.WidthFunction = 0.5,
    offset: float | typings.OffsetFunction = 0,
    layer: typings.LayerSpec = "WG",
    sections: Sequence[KFactorySectionSpec | Section] | None = None,
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
    name: str | None = None,
    main_section_name: str = "_default",
) -> CrossSection:
    """Return a native kfactory cross-section.

    Args:
        width: main Section width (um) or parameterized function from 0 to 1.
        offset: main Section center offset (um) or parameterized function from 0 to 1.
        layer: main section layer.
        sections: absolute auxiliary strips as ``(layer, section_min,
            section_max)`` tuples.
        port_names: legacy extrusion metadata. It is temporarily ignored.
        port_types: legacy extrusion metadata. It is temporarily ignored.
        bbox_layers: list of layers bounding boxes to extrude.
        bbox_offsets: list of offset from bounding box edge.
        cladding_layers: list of layers to extrude.
        cladding_offsets: offset from main Section edge. Single float is
            broadcast to all cladding layers.
        cladding_simplify: legacy extrusion metadata. It is temporarily ignored.
        cladding_centers: center offset for each cladding layer. Defaults to 0. \
                Single float is broadcast to all cladding layers.
        radius: routing bend radius (um).
        radius_min: min acceptable bend radius.
        name: native cross-section name.
        main_section_name: legacy section metadata. It is temporarily ignored.

    Example:
        ```python
        import gdsfactory as gf

        xs = gf.cross_section.cross_section(width=0.5, offset=0, layer='WG')
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()


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
        ```
    """
    if port_names != ("o1", "o2"):
        _warn("port_names is legacy extrusion metadata and is temporarily ignored.")
    if port_types != ("optical", "optical"):
        _warn("port_types is legacy extrusion metadata and is temporarily ignored.")
    if cladding_simplify is not None:
        _warn(
            "cladding_simplify is legacy extrusion metadata and is temporarily ignored."
        )
    if main_section_name != "_default":
        _warn(
            "main_section_name is legacy section metadata and is temporarily "
            "ignored; use name for the native cross-section name."
        )

    native_sections: list[KFactorySectionSpec] = []
    for section in sections or ():
        if isinstance(section, Section):
            native_sections.append(_section_to_kfactory_spec(section, warn=True))
        else:
            native_sections.append(section)

    return kfactory_cross_section(
        width=_nominal_value(width, "width", warn=True),
        offset=_nominal_value(offset, "offset", warn=True),
        layer=layer,
        sections=native_sections,
        bbox_layers=bbox_layers,
        bbox_offsets=bbox_offsets,
        cladding_layers=cladding_layers,
        cladding_offsets=cladding_offsets,
        cladding_centers=cladding_centers,
        radius=radius,
        radius_min=radius_min,
        name=name,
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
                "SymmetricCrossSection",
                "AsymmetricCrossSection",
                "DCrossSection",
                "DAsymmetricCrossSection",
                "LegacyCrossSection",
                "gf.CrossSection",
                "gf.SymmetricCrossSection",
                "gf.AsymmetricCrossSection",
                "gf.DCrossSection",
                "gf.DAsymmetricCrossSection",
                "gf.LegacyCrossSection",
                "gdsfactory.CrossSection",
                "gdsfactory.SymmetricCrossSection",
                "gdsfactory.AsymmetricCrossSection",
                "gdsfactory.DCrossSection",
                "gdsfactory.DAsymmetricCrossSection",
                "gdsfactory.LegacyCrossSection",
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
                    return _is_cross_section_type(resolved_type)

            except (TypeError, AttributeError, ValueError):
                pass  # Ignore type resolution errors

            return False

        # Direct type comparison
        if _is_cross_section_type(return_type):
            return True

        # Check if it's a subclass of a supported cross-section class.
        if isinstance(return_type, type):
            try:
                return _is_cross_section_type(return_type)
            except TypeError:
                # Handle cases where return_type is not a class
                return False

    except Exception as e:
        if verbose:
            logger.warning(f"Error checking cross-section for {name}: {e}")

    return False


def _is_cross_section_type(value: Any) -> bool:
    if value in (
        LegacyCrossSection,
        kf.DCrossSection,
        kf.DAsymmetricCrossSection,
        kf.SymmetricalCrossSection,
        kf.AsymmetricalCrossSection,
        kf.CrossSection,
    ):
        return True
    if isinstance(value, type):
        return issubclass(
            value,
            (
                LegacyCrossSection,
                kf.DCrossSection,
                kf.DAsymmetricCrossSection,
                kf.SymmetricalCrossSection,
                kf.AsymmetricalCrossSection,
            ),
        )
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
