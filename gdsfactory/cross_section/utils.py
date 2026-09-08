"""Factories for kfactory cross sections, expressed in micrometers."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from functools import wraps
from inspect import getmembers, signature
from types import ModuleType
from typing import Any, ParamSpec, Protocol

import kfactory as kf

from gdsfactory import typings
from gdsfactory.config import CONF, ErrorType
from gdsfactory.cross_section.base import CrossSection, CrossSectionFactory, Sections

cross_sections: dict[str, CrossSectionFactory] = {}
P = ParamSpec("P")


class CrossSectionCallable(Protocol[P]):
    __name__: str

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> CrossSection: ...


def xsection[**P](
    func: CrossSectionCallable[P],
    xs_container: dict[str, CrossSectionFactory] = cross_sections,
) -> CrossSectionCallable[P]:
    """Register a profile factory without evaluating it before a PDK is active."""
    sig = signature(func)
    defaults = sig.bind()
    defaults.apply_defaults()

    @wraps(func)
    def factory(*args: P.args, **kwargs: P.kwargs) -> CrossSection:
        xs = func(*args, **kwargs)
        arguments = sig.bind(*args, **kwargs)
        arguments.apply_defaults()
        is_default = all(
            type(value) is type(defaults.arguments[key])
            and value == defaults.arguments[key]
            for key, value in arguments.arguments.items()
        )
        if is_default and not xs.base.is_named:
            base = xs.kcl.get_base_cross_section(
                xs.base.model_copy(update={"name": func.__name__})
            )
            xs = (
                kf.DCrossSection(kcl=xs.kcl, base=base)
                if isinstance(base, kf.SymmetricalCrossSection)
                else kf.DAsymmetricCrossSection(kcl=xs.kcl, base=base)
            )
        return xs

    xs_container[func.__name__] = factory
    return factory


def cross_section(
    width: float | None = 0.5,
    offset: float = 0,
    layer: typings.LayerSpec | kf.kdb.LayerInfo = "WG",
    sections: Sections | None = None,
    bbox_layers: Sequence[typings.LayerSpec | kf.kdb.LayerInfo] | None = None,
    bbox_offsets: typings.Floats | None = None,
    cladding_layers: typings.LayerSpecs | None = None,
    cladding_offsets: float | typings.Floats | None = None,
    cladding_centers: float | typings.Floats | None = None,
    radius: float | None = None,
    radius_min: float | None = None,
    name: str | None = None,
    kcl: kf.KCLayout | None = None,
) -> CrossSection:
    """Build a profile, rounding each signed strip edge with KLayout's DBU rule.

    Auxiliary sections are ``(layer, minimum, maximum)`` tuples in micrometers,
    or kfactory DCrossSectionLayer objects. The main strip stays separate from
    overlapping auxiliary strips. Mirrored auxiliary bands become edge-relative
    enclosures when the main strip is centered; other profiles are asymmetric.
    Port names, dynamic widths and other extrusion options belong to extrude().
    """
    from gdsfactory.pdk import get_layer_info

    kcl = kcl if kcl is not None else kf.kcl

    def strip(
        layer: typings.LayerSpec | kf.kdb.LayerInfo, lo: float, hi: float
    ) -> kf.CrossSectionLayer:
        return kf.CrossSectionLayer(
            layer=get_layer_info(layer),
            section_min=kcl.to_dbu(lo),
            section_max=kcl.to_dbu(hi),
        )

    if width is None:
        if not sections:
            raise ValueError("A main width or a nonempty list of sections is required")
        first, *sections = sections
        main = (
            strip(first.layer, first.section_min, first.section_max)
            if isinstance(first, kf.DCrossSectionLayer)
            else strip(*first)
        )
        width = kcl.to_um(main.width)
    else:
        main = strip(layer, offset - width / 2, offset + width / 2)
    assert width is not None
    auxiliary = [
        strip(s.layer, s.section_min, s.section_max)
        if isinstance(s, kf.DCrossSectionLayer)
        else strip(*s)
        for s in sections or ()
    ]
    if cladding_layers:

        def broadcast(value: float | typings.Floats | None) -> Sequence[float]:
            if value is None:
                return [0.0] * len(cladding_layers)
            if isinstance(value, (int, float)):
                return [value] * len(cladding_layers)
            return value

        for layer_spec, d, center in zip(
            cladding_layers,
            broadcast(cladding_offsets),
            broadcast(cladding_centers),
            strict=True,
        ):
            auxiliary.append(
                strip(layer_spec, center - width / 2 - d, center + width / 2 + d)
            )
    bbox = {
        get_layer_info(layer_spec): kcl.to_dbu(d)
        for layer_spec, d in zip(
            bbox_layers or (),
            bbox_offsets
            if bbox_offsets is not None
            else [0.0] * len(bbox_layers or ()),
            strict=True,
        )
    }
    profile = kf.AsymmetricalCrossSection(
        layer=main.layer,
        section_min=main.section_min,
        section_max=main.section_max,
        sections=tuple(auxiliary),
        bbox_sections=bbox,
        radius=kcl.to_dbu(radius),
        radius_min=kcl.to_dbu(radius_min),
        name=name or "",
    )
    bands = {(s.layer, s.section_min, s.section_max) for s in profile.sections}
    if main.section_min == -main.section_max and bands == {
        (layer, -hi, -lo) for layer, lo, hi in bands
    }:
        half = main.section_max
        enclosure = kf.LayerEnclosure(
            main_layer=main.layer,
            sections=[
                (layer, hi - half) if lo == -hi else (layer, lo - half, hi - half)
                for layer, lo, hi in bands
                if hi > 0
            ],
            bbox_sections=list(bbox.items()),
        )
        base = kcl.get_symmetrical_cross_section(
            kf.SymmetricalCrossSection(
                width=main.width,
                enclosure=enclosure,
                name=name,
                radius=profile.radius,
                radius_min=profile.radius_min,
            )
        )
        return kf.DCrossSection(kcl=kcl, base=base)
    return kf.DAsymmetricCrossSection(
        kcl=kcl, base=kcl.get_asymmetrical_cross_section(profile)
    )


def with_width(xs: CrossSection, width: float) -> CrossSection:
    """Replace the main width while keeping auxiliary strips at absolute bounds."""
    if width == xs.width:
        return xs
    main, *sections = xs.get_sections()
    return cross_section(
        width=width,
        offset=(main.section_min + main.section_max) / 2,
        layer=xs.layer,
        sections=sections,
        bbox_layers=list(xs.bbox_sections),
        bbox_offsets=list(xs.bbox_sections.values()),
        radius=xs.radius,
        radius_min=xs.radius_min,
        kcl=xs.kcl,
    )


def get_port_cross_section(
    width: float,
    layer: typings.LayerSpec | kf.kdb.LayerInfo,
    kcl: kf.KCLayout,
    *,
    offset: float = 0,
) -> CrossSection:
    """Create a port profile using the PDK's explicit per-layer factory.

    Unconfigured layers get a bare profile with no radius metadata. As with any
    profile, its radii cannot be supplied or changed after first registration.
    """
    from gdsfactory.pdk import get_active_pdk, get_layer_info

    layer = get_layer_info(layer)
    factory = get_active_pdk().port_cross_sections.get((layer.layer, layer.datatype))
    return (
        factory(width=width, layer=layer, offset=offset, kcl=kcl)
        if factory is not None
        else cross_section(width=width, offset=offset, layer=layer, kcl=kcl)
    )


def section_cross_section(
    section: kf.DCrossSectionLayer, kcl: kf.KCLayout
) -> tuple[CrossSection, float]:
    """Return a strip's own profile and on-grid transverse origin (um).

    An odd-DBU span uses asymmetric bounds; neither its width nor either edge
    needs rounding again when the strip is placed at the returned origin.
    """
    strip = section.to_itype(kcl)
    half = strip.width // 2
    defaults = get_port_cross_section(
        kcl.to_um(strip.width),
        section.layer,
        kcl,
        offset=kcl.to_um(strip.width % 2) / 2,
    )
    profile = cross_section(
        width=None,
        sections=[(section.layer, kcl.to_um(-half), kcl.to_um(strip.width - half))],
        radius=defaults.radius,
        radius_min=defaults.radius_min,
        kcl=kcl,
    )
    return profile, kcl.to_um(strip.section_min + half)


def add_bbox(
    component: typings.AnyComponentT,
    xs: CrossSection,
    top: float | None = None,
    bottom: float | None = None,
    right: float | None = None,
    left: float | None = None,
) -> typings.AnyComponentT:
    """Add the profile's bounding-box layers around the component."""
    from gdsfactory.add_padding import get_padding_points

    polygons: list[tuple[kf.kdb.LayerInfo, list[typings.Coordinate]]] = [
        (
            layer,
            get_padding_points(
                component=component,
                default=d,
                top=top if top is not None else d,
                bottom=bottom if bottom is not None else d,
                right=right if right is not None else d,
                left=left if left is not None else d,
            ),
        )
        for layer, d in xs.bbox_sections.items()
    ]
    for layer, points in polygons:
        component.add_polygon(points, layer=layer)
    return component


def validate_radius(
    xs: CrossSection, radius: float, error_type: ErrorType | None = None
) -> None:
    """Check a bend against the profile's minimum radius."""
    minimum = xs.radius_min
    if minimum is not None and radius < minimum:
        message = f"min_bend_radius {radius} < CrossSection.radius_min {minimum}."
        error_type = error_type or CONF.bend_radius_error_type
        if error_type == ErrorType.ERROR:
            raise ValueError(message)
        if error_type == ErrorType.WARNING:
            warnings.warn(message, stacklevel=2)


def is_cross_section(name: str, obj: Any, verbose: bool = False) -> bool:
    """Whether an object is a cross-section factory."""
    if name.startswith("_") or not callable(obj):
        return False
    if cross_sections.get(name) is obj:
        return True
    try:
        annotation = signature(obj).return_annotation
    except (TypeError, ValueError):
        return False
    return annotation is CrossSection or annotation in (
        "CrossSection",
        "gf.CrossSection",
        "gdsfactory.CrossSection",
    )


def get_cross_sections(
    modules: Sequence[ModuleType] | ModuleType, verbose: bool = False
) -> dict[str, CrossSectionFactory]:
    """Collect profile factories from modules."""
    modules = [modules] if isinstance(modules, ModuleType) else modules
    return {
        name: obj
        for module in modules
        for name, obj in getmembers(module)
        if is_cross_section(name, obj, verbose)
    }
