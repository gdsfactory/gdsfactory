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
from gdsfactory.cross_section.base import (
    CrossSection,
    CrossSectionFactory,
    Section,
    Sections,
    SectionSpec,
)

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
    port_names: tuple[str | None, str | None] | None = None,
    port_types: str | tuple[str, str] | None = None,
    bbox_layers: Sequence[typings.LayerSpec | kf.kdb.LayerInfo] | None = None,
    bbox_offsets: typings.Floats | None = None,
    cladding_layers: typings.LayerSpecs | None = None,
    cladding_offsets: float | typings.Floats | None = None,
    cladding_simplify: float | typings.Floats | None = None,
    cladding_centers: float | typings.Floats | None = None,
    radius: float | None = None,
    radius_min: float | None = None,
    name: str | None = None,
    kcl: kf.KCLayout | None = None,
) -> CrossSection:
    """Build a profile, rounding each signed strip edge with KLayout's DBU rule.

    Auxiliary sections are ``(layer, minimum, maximum)`` tuples in micrometers,
    kfactory DCrossSectionLayer objects or ``Section`` objects. The main strip
    stays separate from overlapping auxiliary strips. Mirrored auxiliary bands
    become edge-relative enclosures when the main strip is centered; other
    profiles are asymmetric.

    Port names and types of the main strip, ``Section`` extrusion settings and
    ``cladding_simplify`` are stored once per profile geometry and used by every
    extrusion of the profile. Without them, extrusions use the PDK's port
    conventions. Dynamic widths and offsets belong to extrude().

    Args:
        width: main strip width (um). None takes the main strip from the first
            section.
        offset: center of the main strip (um).
        layer: main strip layer.
        sections: auxiliary strips.
        port_names: (start, end) port names of the main strip.
        port_types: port type of the main strip, or (start, end) port types.
        bbox_layers: layers of the bounding box sections.
        bbox_offsets: offsets of the bounding box sections (um).
        cladding_layers: layers of strips around the main strip.
        cladding_offsets: distance of each cladding beyond the main strip
            edges (um). A single value applies to all claddings.
        cladding_simplify: simplification tolerance of each cladding (um). A
            single value applies to all claddings.
        cladding_centers: center of each cladding (um). A single value applies
            to all claddings.
        radius: routing bend radius (um).
        radius_min: minimum bend radius (um).
        name: name of the profile.
        kcl: layout to register the profile in.
    """
    from gdsfactory.pdk import get_layer_info

    kcl = kcl if kcl is not None else kf.kcl
    main_settings = (
        Section(layer, 0, 0, port_names=port_names, port_types=port_types)
        if port_names is not None or port_types is not None
        else None
    )
    settings: list[tuple[kf.CrossSectionLayer, Section]] = []

    def strip(
        layer: typings.LayerSpec | kf.kdb.LayerInfo, lo: float, hi: float
    ) -> kf.CrossSectionLayer:
        return kf.CrossSectionLayer(
            layer=get_layer_info(layer),
            section_min=kcl.to_dbu(lo),
            section_max=kcl.to_dbu(hi),
        )

    def section_strip(section: SectionSpec) -> kf.CrossSectionLayer:
        if isinstance(section, Section | kf.DCrossSectionLayer):
            return strip(section.layer, section.section_min, section.section_max)
        return strip(*section)

    if width is None:
        if not sections:
            raise ValueError("A main width or a nonempty list of sections is required")
        first, *sections = sections
        main = section_strip(first)
        if isinstance(first, Section) and first.has_extrude_settings():
            if main_settings is not None:
                raise ValueError(
                    "Set the main strip's ports either with port_names/port_types "
                    "or with the first section, not both."
                )
            main_settings = first
        width = kcl.to_um(main.width)
    else:
        main = strip(layer, offset - width / 2, offset + width / 2)
    assert width is not None
    auxiliary = []
    for section in sections or ():
        auxiliary.append(section_strip(section))
        if isinstance(section, Section) and section.has_extrude_settings():
            settings.append((auxiliary[-1], section))
    if cladding_layers:

        def broadcast(
            value: float | typings.Floats | None, default: float | None
        ) -> Sequence[float | None]:
            if value is None:
                return [default] * len(cladding_layers)
            if isinstance(value, (int, float)):
                return [value] * len(cladding_layers)
            return value

        for layer_spec, d, center, tolerance in zip(
            cladding_layers,
            broadcast(cladding_offsets, 0.0),
            broadcast(cladding_centers, 0.0),
            broadcast(cladding_simplify, None),
            strict=True,
        ):
            assert d is not None and center is not None
            auxiliary.append(
                strip(layer_spec, center - width / 2 - d, center + width / 2 + d)
            )
            if tolerance is not None:
                settings.append(
                    (auxiliary[-1], Section(layer_spec, 0, 0, simplify=tolerance))
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
        xs: CrossSection = kf.DCrossSection(kcl=kcl, base=base)
    else:
        xs = kf.DAsymmetricCrossSection(
            kcl=kcl, base=kcl.get_asymmetrical_cross_section(profile)
        )
    if main_settings is not None or settings:
        set_extrude_config(xs, _extrude_config(xs, main_settings, settings))
    return xs


def _extrude_config(
    xs: CrossSection,
    main: Section | None,
    auxiliary: Sequence[tuple[kf.CrossSectionLayer, Section]],
) -> kf.ExtrudeConfig:
    """Convert per-strip settings to a config keyed by get_sections() index."""
    strips = [
        (s.layer.layer, s.layer.datatype, s.section_min, s.section_max)
        for s in xs.base.get_sections()
    ]
    by_index = {0: main} if main is not None else {}
    for strip, section in auxiliary:
        key = (
            strip.layer.layer,
            strip.layer.datatype,
            strip.section_min,
            strip.section_max,
        )
        index = next((i for i, s in enumerate(strips) if i and s == key), None)
        if index is None:
            raise ValueError(
                f"{section!r} overlaps another strip on its layer and was merged "
                "with it, so it can't have its own extrusion settings."
            )
        other = by_index.get(index)
        if other is not None and other.extrude_settings() != section.extrude_settings():
            raise ValueError(
                f"{section!r} and {other!r} are the same strip but have different "
                "extrusion settings."
            )
        by_index[index] = section

    ports = None
    if any(
        s.port_names is not None or s.port_type is not None for s in by_index.values()
    ):
        ports = default_extrude_ports(xs)
        for i, s in by_index.items():
            if s.port_names is None and s.port_type is None:
                continue
            if s.port_names is None and i not in ports:
                raise ValueError(
                    f"{s!r} has no default ports, port_types needs port_names."
                )
            start, end, port_type = ports.get(i, (None, None, "optical"))
            if s.port_names is not None:
                start, end = s.port_names
            ports[i] = (start, end, s.port_type or port_type)

    kcl = xs.kcl
    simplify = {
        i: kcl.to_dbu(s.simplify) for i, s in by_index.items() if s.simplify is not None
    }
    return kf.ExtrudeConfig(
        ports=ports,
        insets={
            i: (kcl.to_dbu(s.insets[0]), kcl.to_dbu(s.insets[1]))
            for i, s in by_index.items()
            if s.insets is not None
        },
        simplify=simplify or None,
        hidden=frozenset(i for i, s in by_index.items() if s.hidden),
    )


def set_extrude_config(xs: CrossSection, config: kf.ExtrudeConfig) -> None:
    """Store the extrusion settings of a profile's geometry once.

    Raises:
        kfactory.exceptions.ExtrudeConfigConflictError: if the geometry already
            has different extrusion settings.
    """
    if xs.kcl.get_extrude_config(xs) != config:
        xs.kcl.set_extrude_config(xs, config, on_conflict="raise")


def default_extrude_ports(
    xs: CrossSection,
) -> dict[int, tuple[str | None, str | None, str]]:
    """Ports of a profile's strips by the PDK's port type conventions.

    The main strip's port type comes from ``Pdk.layer_port_types`` and auxiliary
    strips' from ``Pdk.auxiliary_port_types``; strips without a type get no
    ports. Names count up per prefix: ``o1``/``o2`` for optical ports and
    ``e1``/``e2`` for electrical ones.
    """
    from gdsfactory.pdk import get_active_pdk

    pdk = get_active_pdk()
    result: dict[int, tuple[str | None, str | None, str]] = {}
    counters: dict[str, int] = {}
    main_layer = (xs.layer.layer, xs.layer.datatype)
    for i, s in enumerate(xs.get_sections()):
        layer = (s.layer.layer, s.layer.datatype)
        port_type = (
            pdk.layer_port_types.get(layer, "optical")
            if i == 0
            else pdk.auxiliary_port_types.get((main_layer, layer))
        )
        if port_type is None:
            continue
        prefix = "e" if port_type == "electrical" else "o"
        n = counters.get(prefix, 0) + 1
        result[i] = (f"{prefix}{n}", f"{prefix}{n + 1}", port_type)
        counters[prefix] = n + 1
    return result


def with_width(xs: CrossSection, width: float) -> CrossSection:
    """Replace the main width while keeping auxiliary strips at absolute bounds.

    The profile's extrusion settings carry over to the new geometry.
    """
    if width == xs.width:
        return xs
    main, *sections = xs.get_sections()
    resized = cross_section(
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
    config = xs.kcl.get_extrude_config(xs)
    if config is not None:
        set_extrude_config(resized, config)
    return resized


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
