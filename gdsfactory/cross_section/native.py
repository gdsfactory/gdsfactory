"""Native kfactory cross-section construction helpers.

The helpers in this module describe only the geometric profile.  Port metadata
and other extrusion options deliberately remain outside the native kfactory
cross-section until the extrusion migration is complete.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from numbers import Real
from typing import cast

import kfactory as kf

from gdsfactory import typings

type NativeSection = tuple[typings.LayerSpec, float, float] | kf.DCrossSectionLayer
type NativeCrossSection = kf.DCrossSection | kf.DAsymmetricCrossSection
type _DbuSection = tuple[kf.kdb.LayerInfo, int, int]


def _layer_info(layer: typings.LayerSpec | kf.kdb.LayerInfo) -> kf.kdb.LayerInfo:
    if isinstance(layer, kf.kdb.LayerInfo):
        return layer

    # Import lazily to avoid the cross_section -> pdk -> cross_section cycle.
    from gdsfactory.pdk import get_layer_info

    return get_layer_info(layer)


def _layer_key(layer: kf.kdb.LayerInfo) -> tuple[int, int]:
    return layer.layer, layer.datatype


def _merge_sections(sections: Sequence[_DbuSection]) -> tuple[_DbuSection, ...]:
    """Merge touching/overlapping sections on the same physical layer."""
    by_layer: dict[tuple[int, int], list[_DbuSection]] = defaultdict(list)
    for section in sections:
        by_layer[_layer_key(section[0])].append(section)

    merged: list[_DbuSection] = []
    for layer_sections in by_layer.values():
        layer_sections.sort(key=lambda section: (section[1], section[2]))
        layer = layer_sections[0][0]
        section_min = layer_sections[0][1]
        section_max = layer_sections[0][2]
        for _, next_min, next_max in layer_sections[1:]:
            if next_min <= section_max:
                section_max = max(section_max, next_max)
            else:
                merged.append((layer, section_min, section_max))
                section_min = next_min
                section_max = next_max
        merged.append((layer, section_min, section_max))

    return tuple(
        sorted(
            merged,
            key=lambda section: (
                section[0].layer,
                section[0].datatype,
                section[1],
                section[2],
            ),
        )
    )


def _is_symmetric(sections: Sequence[_DbuSection]) -> bool:
    normalized = _merge_sections(sections)
    section_set = {
        (_layer_key(layer), section_min, section_max)
        for layer, section_min, section_max in normalized
    }
    return section_set == {
        (layer_key, -section_max, -section_min)
        for layer_key, section_min, section_max in section_set
    }


def _to_enclosure_sections(
    sections: Sequence[_DbuSection], half_width: int, kcl: kf.KCLayout
) -> list[tuple[kf.kdb.LayerInfo, float] | tuple[kf.kdb.LayerInfo, float, float]]:
    """Convert absolute symmetric strips into edge-relative enclosures."""
    enclosure_sections: list[
        tuple[kf.kdb.LayerInfo, float] | tuple[kf.kdb.LayerInfo, float, float]
    ] = []

    for layer, section_min, section_max in _merge_sections(sections):
        # A symmetric interval crossing the center line is represented by a
        # d_max-only enclosure.  The remaining intervals are positive-side
        # bands; their mirrored negative-side bands are implicit.
        if section_min == -section_max:
            enclosure_sections.append((layer, kcl.to_um(section_max - half_width)))
        elif section_min >= 0:
            enclosure_sections.append(
                (
                    layer,
                    kcl.to_um(section_min - half_width),
                    kcl.to_um(section_max - half_width),
                )
            )
        elif section_max <= 0:
            continue
        else:
            raise ValueError(
                "A symmetric cross section has an auxiliary strip crossing the "
                "center line without being centered."
            )

    return enclosure_sections


def _normalize_native_sections(
    sections: Sequence[NativeSection], kcl: kf.KCLayout
) -> tuple[_DbuSection, ...]:
    normalized: list[_DbuSection] = []
    for section in sections:
        if isinstance(section, kf.DCrossSectionLayer):
            layer = section.layer
            section_min = kcl.to_dbu(section.section_min)
            section_max = kcl.to_dbu(section.section_max)
        else:
            layer_spec, section_min_um, section_max_um = section
            layer = _layer_info(layer_spec)
            section_min = kcl.to_dbu(section_min_um)
            section_max = kcl.to_dbu(section_max_um)

        if section_min >= section_max:
            raise ValueError(
                "Native cross-section sections require section_min < section_max "
                f"after grid snapping, got {section_min=} and {section_max=} for "
                f"layer {layer}."
            )
        normalized.append((layer, section_min, section_max))

    return tuple(normalized)


def native_cross_section(
    width: float,
    offset: float = 0,
    layer: typings.LayerSpec = "WG",
    sections: Sequence[NativeSection] | None = None,
    bbox_layers: typings.LayerSpecs | None = None,
    bbox_offsets: typings.Floats | None = None,
    cladding_layers: typings.LayerSpecs | None = None,
    cladding_offsets: float | typings.Floats | None = None,
    cladding_centers: float | typings.Floats | None = None,
    radius: float | None = 10.0,
    radius_min: float | None = 7.0,
    name: str | None = None,
    kcl: kf.KCLayout | None = None,
) -> NativeCrossSection:
    """Construct a native kfactory cross-section from µm geometry.

    ``sections`` contains auxiliary absolute strips as
    ``(layer, section_min, section_max)`` tuples, measured from the path
    centerline.  The main strip is defined by ``width``, ``offset``, and
    ``layer``.  The returned object is a µm-based kfactory wrapper.

    This function intentionally has no port, transition, or dynamic extrusion
    arguments.  Those properties are gdsfactory extrusion metadata and will be
    handled by the migration's extrusion specification.
    """
    target_kcl = kcl if kcl is not None else kf.kcl
    main_layer = _layer_info(layer)
    main_min = target_kcl.to_dbu(offset - width / 2)
    main_max = target_kcl.to_dbu(offset + width / 2)
    if main_min >= main_max:
        raise ValueError(
            "Native cross-section requires a positive main width after grid "
            f"snapping, got {main_min=} and {main_max=} for {width=} and {offset=}."
        )

    auxiliary_sections: list[NativeSection] = list(sections or ())
    if cladding_layers:
        if isinstance(cladding_layers, (str, bytes)):
            raise TypeError("cladding_layers must be a sequence of layer specs.")

        def _broadcast(
            value: float | typings.Floats | None, default: float
        ) -> list[float]:
            if isinstance(value, Real):
                return [float(value)] * len(cladding_layers)
            if value is None:
                return [default] * len(cladding_layers)
            if not isinstance(value, Sequence):
                raise TypeError(
                    "cladding_offsets and cladding_centers must be a number or "
                    "a sequence of numbers."
                )
            if len(value) != len(cladding_layers):
                raise ValueError(
                    "cladding_layers, cladding_offsets, and cladding_centers must "
                    "have the same length."
                )
            return [float(item) for item in cast(Sequence[float | int], value)]

        offsets = _broadcast(cladding_offsets, 0)
        centers = _broadcast(cladding_centers, 0)
        if len(cladding_layers) != len(offsets):
            raise ValueError(
                "cladding_layers, cladding_offsets, and cladding_centers must "
                "have the same length."
            )
        auxiliary_sections.extend(
            (
                cladding_layer,
                center - (width / 2 + cladding_offset),
                center + (width / 2 + cladding_offset),
            )
            for cladding_layer, cladding_offset, center in zip(
                cladding_layers, offsets, centers, strict=True
            )
        )

    main_section = (main_layer, main_min, main_max)
    auxiliary_dbu = _normalize_native_sections(auxiliary_sections, target_kcl)
    all_sections = (main_section, *auxiliary_dbu)

    if bbox_layers is None:
        resolved_bbox_layers: list[kf.kdb.LayerInfo] = []
        resolved_bbox_offsets: list[float] = []
    else:
        resolved_bbox_layers = [_layer_info(bbox_layer) for bbox_layer in bbox_layers]
        if bbox_offsets is None:
            resolved_bbox_offsets = [0.0] * len(resolved_bbox_layers)
        else:
            resolved_bbox_offsets = [float(item) for item in bbox_offsets]
            if len(resolved_bbox_offsets) != len(resolved_bbox_layers):
                raise ValueError(
                    "bbox_layers and bbox_offsets must have the same length."
                )

    if _is_symmetric(all_sections):
        if main_min != -main_max:
            raise ValueError(
                "The main section is not centered, so the native profile must be "
                "asymmetric."
            )
        enclosure_sections = _to_enclosure_sections(auxiliary_dbu, main_max, target_kcl)
        return kf.DCrossSection(
            kcl=target_kcl,
            width=target_kcl.to_um(main_max - main_min),
            layer=main_layer,
            sections=enclosure_sections,
            bbox_layers=resolved_bbox_layers,
            bbox_offsets=resolved_bbox_offsets,
            radius=radius,
            radius_min=radius_min,
            name=name,
        )

    return kf.DAsymmetricCrossSection(
        kcl=target_kcl,
        layer=main_layer,
        section_min=target_kcl.to_um(main_min),
        section_max=target_kcl.to_um(main_max),
        sections=tuple(
            kf.DCrossSectionLayer(
                layer=section_layer,
                section_min=target_kcl.to_um(section_min),
                section_max=target_kcl.to_um(section_max),
            )
            for section_layer, section_min, section_max in auxiliary_dbu
        ),
        bbox_sections={
            _layer_info(bbox_layer): float(bbox_offset)
            for bbox_layer, bbox_offset in zip(
                resolved_bbox_layers, resolved_bbox_offsets, strict=True
            )
        },
        radius=radius,
        radius_min=radius_min,
        name=name,
    )


__all__ = ["NativeCrossSection", "NativeSection", "native_cross_section"]
