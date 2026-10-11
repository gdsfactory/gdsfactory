"""Preset cross-section factory functions.

Includes strip, rib, slot, nitride, metal, and other standard waveguide
cross-sections.
"""

from __future__ import annotations

from typing import Any

from gdsfactory import typings
from gdsfactory.cross_section.base import (
    CrossSection,
    Sections,
    SectionSpec,
)
from gdsfactory.cross_section.utils import cross_section, xsection

radius_nitride = 20
radius_rib = 20


@xsection
def strip(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    radius: float = 10.0,
    radius_min: float = 3.5,
    **kwargs: Any,
) -> CrossSection:
    """Return Strip cross_section.

    Args:
        width: main strip width (um).
        layer: main section layer.
        radius: routing bend radius (um).
        radius_min: min acceptable bend radius.
        kwargs: cross_section settings.
    """
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        **kwargs,
    )


@xsection
def rib(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    radius: float = radius_rib,
    radius_min: float | None = 7,
    cladding_layers: typings.LayerSpecs = ("SLAB90",),
    cladding_offsets: typings.Floats = (3,),
    **kwargs: Any,
) -> CrossSection:
    """Return Rib cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        cladding_layers=cladding_layers,
        cladding_offsets=cladding_offsets,
        **kwargs,
    )


@xsection
def rib_bbox(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    radius: float = radius_rib,
    radius_min: float | None = None,
    bbox_layers: typings.LayerSpecs = ("SLAB90",),
    bbox_offsets: typings.Floats = (3,),
    **kwargs: Any,
) -> CrossSection:
    """Return Rib cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        bbox_layers=bbox_layers,
        bbox_offsets=bbox_offsets,
        **kwargs,
    )


@xsection
def rib2(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    layer_slab: typings.LayerSpec = "SLAB90",
    radius: float = radius_rib,
    radius_min: float | None = None,
    width_slab: float = 6,
    **kwargs: Any,
) -> CrossSection:
    """Return Rib cross_section."""
    sections = ((layer_slab, -(width_slab / 2), width_slab / 2),)
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        sections=sections,
        **kwargs,
    )


@xsection
def nitride(
    width: float = 1.0,
    layer: typings.LayerSpec = "WGN",
    radius: float = radius_nitride,
    radius_min: float | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Return Strip cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        **kwargs,
    )


@xsection
def strip_rib_tip(
    width: float = 0.5,
    width_tip: float = 0.2,
    layer: typings.LayerSpec = "WG",
    layer_slab: typings.LayerSpec = "SLAB90",
    radius: float = 10.0,
    radius_min: float | None = 5,
    **kwargs: Any,
) -> CrossSection:
    """Return Rib tip cross_section."""
    sections = ((layer_slab, -(width_tip / 2), width_tip / 2),)
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        sections=sections,
        **kwargs,
    )


@xsection
def strip_nitride_tip(
    width: float = 1.0,
    layer: typings.LayerSpec = "WGN",
    layer_silicon: typings.LayerSpec = "WG",
    width_tip_nitride: float = 0.2,
    width_tip_silicon: float = 0.1,
    radius: float = radius_nitride,
    radius_min: float | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Return the end of the nitride tip.

    Args:
        width: main strip width (um).
        layer: main section layer.
        layer_silicon: silicon layer.
        width_tip_nitride: in um.
        width_tip_silicon: in um.
        radius: routing bend radius (um).
        radius_min: min acceptable bend radius.
        kwargs: cross_section settings.

    """
    sections = (
        (layer, -(width_tip_nitride / 2), width_tip_nitride / 2),
        (layer_silicon, -(width_tip_silicon / 2), width_tip_silicon / 2),
    )
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        sections=sections,
        **kwargs,
    )


@xsection
def slot(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG_ABSTRACT",
    slot_width: float = 0.04,
    rail_layer: typings.LayerSpec = "WG",
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Return CrossSection Slot (with an etched region in the center).

    Args:
        width: main strip width (um).
        layer: main section layer.
        slot_width: in um.
        rail_layer: rail layer.
        sections: list of (layer, minimum, maximum) strips.
        kwargs: other cross section parameters.

    Example:
        ```python
        import gdsfactory as gf

        xs = gf.cross_section.slot(width=0.5, slot_width=0.05, layer='WG')
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
        ```
    """
    if slot_width >= width:
        raise ValueError(f"{width=} must be greater than {slot_width=}")

    rail_width = (width - slot_width) / 2
    rail_offset = (rail_width + slot_width) / 2

    section_list: list[SectionSpec] = list(sections or [])
    section_list.extend(
        [
            (
                rail_layer,
                rail_offset - rail_width / 2,
                rail_offset + rail_width / 2,
            ),
            (
                rail_layer,
                -rail_offset - rail_width / 2,
                -rail_offset + rail_width / 2,
            ),
        ]
    )

    return strip(
        width=width,
        layer=layer,
        sections=section_list,
        **kwargs,
    )


@xsection
def rib_with_trenches(
    width: float = 0.5,
    width_trench: float = 2.0,
    slab_offset: float | None = 0.3,
    width_slab: float | None = None,
    layer: typings.LayerSpec = "WG",
    layer_trench: typings.LayerSpec = "DEEP_ETCH",
    wg_marking_layer: typings.LayerSpec = "WG_ABSTRACT",
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Return CrossSection of rib waveguide defined by trenches.

    Args:
        width: main strip width (um).
        width_trench: in um.
        slab_offset: from the edge of the trench to the edge of the slab.
        width_slab: in um.
        layer: slab layer.
        layer_trench: layer to etch trenches.
        wg_marking_layer: layer to draw over the actual waveguide. \
                This can be useful for booleans, routing, placement ...
        sections: list of (layer, minimum, maximum) strips.
        kwargs: cross_section settings.

                        ┌─────────┐
                        │         │ wg_marking_layer
                        └─────────┘

               ┌────────┐         ┌────────┐
               │        │         │        │layer_trench
               └────────┘         └────────┘

         ┌─────────────────────────────────────────┐
         │                                  layer  │
         │                                         │
         └─────────────────────────────────────────┘
                        ◄─────────►
                           width
         ┌─────┐         ┌────────┐        ┌───────┐
         │     │         │        │        │       │
         │     └─────────┘        └────────┘       │
         │     ◄---------►         ◄-------►       │
         └─────────────────────────────────────────┘
                                            slab_offset
              width_trench                  ──────►
                                                   |
         ◄────────────────────────────────────────►
                      width_slab


    Example:
        ```python
        import gdsfactory as gf

        xs = gf.cross_section.rib_with_trenches(width=0.5)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
        ```
    """
    if slab_offset is None and width_slab is None:
        raise ValueError("Must specify either slab_offset or width_slab")

    if slab_offset is not None and width_slab is not None:
        raise ValueError("Cannot specify both slab_offset and width_slab")

    if slab_offset is not None:
        width_slab = width + 2 * width_trench + 2 * slab_offset

    trench_offset = width / 2 + width_trench / 2
    section_list: list[SectionSpec] = list(sections or ())
    assert width_slab is not None
    section_list.append((layer, -(width_slab / 2), width_slab / 2))
    section_list += [
        (layer_trench, offset - width_trench / 2, offset + width_trench / 2)
        for i, offset in enumerate([+trench_offset, -trench_offset])
    ]

    return cross_section(
        layer=wg_marking_layer,
        width=width,
        sections=tuple(section_list),
        **kwargs,
    )


@xsection
def l_with_trenches(
    width: float = 0.5,
    width_trench: float = 2.0,
    width_slab: float = 7.0,
    layer: typings.LayerSpec = "WG",
    layer_slab: typings.LayerSpec = "WG",
    layer_trench: typings.LayerSpec = "DEEP_ETCH",
    mirror: bool = False,
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Return CrossSection of l waveguide defined by trenches.

    Args:
        width: main strip width (um).
        width_trench: in um.
        width_slab: in um.
        layer: ridge layer. None adds only ridge.
        layer_slab: slab layer.
        layer_trench: layer to etch trenches.
        mirror: this cross section is not symmetric and you can switch orientation.
        sections: list of (layer, minimum, maximum) strips.
        kwargs: cross_section settings.
                          x = 0
                           |
                           |
        _____         __________
             |        |         |
             |________|         |

    ```text
       _________________________
             <------->          |
            width_trench
                       <-------->
                          width
                                |
       <------------------------>
            width_slab
    ```



    Example:
        ```python
        import gdsfactory as gf

        xs = gf.cross_section.l_with_trenches(width=0.5)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
        ```
    """
    mult = 1 if mirror else -1
    trench_offset = mult * (width / 2 + width_trench / 2)
    section_list: list[SectionSpec] = list(sections or ())
    section_list += [
        (
            layer_slab,
            mult * (width_slab / 2 - width / 2) - width_slab / 2,
            mult * (width_slab / 2 - width / 2) + width_slab / 2,
        )
    ]
    section_list += [
        (
            layer_trench,
            trench_offset - width_trench / 2,
            trench_offset + width_trench / 2,
        )
    ]

    return cross_section(
        width=width,
        layer=layer,
        sections=tuple(section_list),
        **kwargs,
    )


@xsection
def metal1(
    width: float = 10,
    layer: typings.LayerSpec = "M1",
    radius: float | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    radius = radius or width
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        **kwargs,
    )


@xsection
def metal2(
    width: float = 10,
    layer: typings.LayerSpec = "M2",
    radius: float | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    radius = radius or width
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        **kwargs,
    )


@xsection
def metal3(
    width: float = 10,
    layer: typings.LayerSpec = "M3",
    radius: float | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    radius = radius or width
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        **kwargs,
    )


@xsection
def gs(
    trace_width: float = 140,
    layer: typings.LayerSpec = "M3",
    gap: float = 120,
    layer_port: typings.LayerSpec = "M3_ABSTRACT",
    radius: float | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Return Ground-Signal-Ground cross_section.

    Args:
        trace_width: in um.
        layer: metal layer.
        gap: between metal lines in um.
        layer_port: port layer.
        radius: bend radius. Optional, defaults to 2*width+gap.
        kwargs: cross_section settings. (ignored)
    """
    width = trace_width
    sections = [
        (layer_port, -(gap / 2), gap / 2),
        (
            layer,
            gap / 2 + width / 2 - width / 2,
            gap / 2 + width / 2 + width / 2,
        ),
        (
            layer,
            -gap / 2 - width / 2 - width / 2,
            -gap / 2 - width / 2 + width / 2,
        ),
    ]
    return cross_section(
        width=None, sections=tuple(sections), radius=radius or 2 * width + gap
    )


@xsection
def gsg(
    trace_width: float = 140,
    layer: typings.LayerSpec = "M3",
    gap: float = 100,
    radius: float | None = None,
) -> CrossSection:
    """Return Ground-Signal-Ground cross_section.

    Args:
        trace_width: in um.
        layer: metal layer.
        gap: between metal lines in um.
        layer_port: port layer.
        radius: bend radius. Optional, defaults to 3*width+2*gap.
    """
    width = trace_width
    sections = [
        (layer, -(width / 2), width / 2),
        (layer, -gap - width - width / 2, -gap - width + width / 2),
        (layer, gap + width - width / 2, gap + width + width / 2),
    ]
    return cross_section(
        width=None, sections=tuple(sections), radius=radius or 3 * width + 2 * gap
    )


metal_routing = metal3


@xsection
def heater_metal(
    width: float = 2.5,
    layer: typings.LayerSpec = "HEATER",
    radius: float | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    radius = radius or width
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        **kwargs,
    )


@xsection
def npp(
    width: float = 0.5,
    layer: typings.LayerSpec = "NPP",
    radius: float | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Return Doped NPP cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        **kwargs,
    )
