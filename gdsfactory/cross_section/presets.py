"""Preset cross-section factory functions.

Includes strip, rib, slot, nitride, metal, and other standard waveguide
cross-sections.
"""

from __future__ import annotations

from typing import Any

from gdsfactory import typings
from gdsfactory.cross_section.base import (
    CrossSection,
    Section,
    Sections,
    nm,
    port_names_electrical,
    port_types_electrical,
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
        width: main Section width (um).
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
def strip_no_ports(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    radius: float = 10.0,
    radius_min: float = 5,
    port_names: typings.IOPorts = ("", ""),
    **kwargs: Any,
) -> CrossSection:
    """Return Strip cross_section without ports.

    Args:
        width: main Section width (um).
        layer: main section layer.
        radius: routing bend radius (um).
        radius_min: min acceptable bend radius.
        port_names: for input and output ('o1', 'o2').
        kwargs: cross_section settings.
    """
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        port_names=port_names,
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
    cladding_simplify: typings.Floats = (50 * nm,),
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
        cladding_simplify=cladding_simplify,
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
    sections = (
        Section(width=width_slab, layer=layer_slab, name="slab", simplify=50 * nm),
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
    sections = (Section(width=width_tip, layer=layer_slab, name="slab"),)
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
        width: main Section width (um).
        layer: main section layer.
        layer_silicon: silicon layer.
        width_tip_nitride: in um.
        width_tip_silicon: in um.
        radius: routing bend radius (um).
        radius_min: min acceptable bend radius.
        kwargs: cross_section settings.

    """
    sections = (
        Section(width=width_tip_nitride, layer=layer, name="tip_nitride"),
        Section(width=width_tip_silicon, layer=layer_silicon, name="tip_silicon"),
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
        width: main Section width (um) or function parameterized from 0 to 1. \
                the width at t==0 is the width at the beginning of the Path. \
                the width at t==1 is the width at the end.
        layer: main section layer.
        slot_width: in um.
        rail_layer: rail layer.
        sections: list of Sections(width, offset, layer, ports).
        kwargs: other cross section parameters.

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.slot(width=0.5, slot_width=0.05, layer='WG')
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    if slot_width >= width:
        raise ValueError(f"{width=} must be greater than {slot_width=}")

    rail_width = (width - slot_width) / 2
    rail_offset = (rail_width + slot_width) / 2

    section_list: list[Section] = list(sections or [])
    section_list.extend(
        [
            Section(
                width=rail_width,
                offset=+rail_offset,
                layer=rail_layer,
                port_names=("o3", "o4"),
                name="left_rail",
            ),
            Section(
                width=rail_width,
                offset=-rail_offset,
                layer=rail_layer,
                port_names=("o5", "o6"),
                name="right_rail",
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
    simplify_slab: float | None = None,
    layer: typings.LayerSpec = "WG",
    layer_trench: typings.LayerSpec = "DEEP_ETCH",
    wg_marking_layer: typings.LayerSpec = "WG_ABSTRACT",
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Return CrossSection of rib waveguide defined by trenches.

    Args:
        width: main Section width (um) or function parameterized from 0 to 1. \
                the width at t==0 is the width at the beginning of the Path. \
                the width at t==1 is the width at the end.
        width_trench: in um.
        slab_offset: from the edge of the trench to the edge of the slab.
        width_slab: in um.
        simplify_slab: Optional Tolerance value for the simplification algorithm. \
                All points that can be removed without changing the resulting\
                polygon by more than the value listed here will be removed.
        layer: slab layer.
        layer_trench: layer to etch trenches.
        wg_marking_layer: layer to draw over the actual waveguide. \
                This can be useful for booleans, routing, placement ...
        sections: list of Sections(width, offset, layer, ports).
        kwargs: cross_section settings.

    .. code::

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


    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.rib_with_trenches(width=0.5)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    if slab_offset is None and width_slab is None:
        raise ValueError("Must specify either slab_offset or width_slab")

    if slab_offset is not None and width_slab is not None:
        raise ValueError("Cannot specify both slab_offset and width_slab")

    if slab_offset is not None:
        width_slab = width + 2 * width_trench + 2 * slab_offset

    trench_offset = width / 2 + width_trench / 2
    section_list: list[Section] = list(sections or ())
    assert width_slab is not None
    section_list.append(
        Section(width=width_slab, layer=layer, name="slab", simplify=simplify_slab)
    )
    section_list += [
        Section(
            width=width_trench, offset=offset, layer=layer_trench, name=f"trench_{i}"
        )
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
        width: main Section width (um) or function parameterized from 0 to 1. \
                the width at t==0 is the width at the beginning of the Path. \
                the width at t==1 is the width at the end.
        width_trench: in um.
        width_slab: in um.
        layer: ridge layer. None adds only ridge.
        layer_slab: slab layer.
        layer_trench: layer to etch trenches.
        mirror: this cross section is not symmetric and you can switch orientation.
        sections: list of Sections(width, offset, layer, ports).
        kwargs: cross_section settings.


    .. code::
                          x = 0
                           |
                           |
        _____         __________
             |        |         |
             |________|         |

       _________________________
             <------->          |
            width_trench
                       <-------->
                          width
                                |
       <------------------------>
            width_slab



    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.l_with_trenches(width=0.5)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    mult = 1 if mirror else -1
    trench_offset = mult * (width / 2 + width_trench / 2)
    section_list: list[Section] = list(sections or ())
    section_list += [
        Section(
            width=width_slab,
            layer=layer_slab,
            offset=mult * (width_slab / 2 - width / 2),
        )
    ]
    section_list += [
        Section(width=width_trench, offset=trench_offset, layer=layer_trench)
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
    port_names: typings.IOPorts = port_names_electrical,
    port_types: typings.IOPorts = port_types_electrical,
    **kwargs: Any,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    radius = radius or width
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
        **kwargs,
    )


@xsection
def metal2(
    width: float = 10,
    layer: typings.LayerSpec = "M2",
    radius: float | None = None,
    port_names: typings.IOPorts = port_names_electrical,
    port_types: typings.IOPorts = port_types_electrical,
    **kwargs: Any,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    radius = radius or width
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
        **kwargs,
    )


@xsection
def metal3(
    width: float = 10,
    layer: typings.LayerSpec = "M3",
    radius: float | None = None,
    port_names: typings.IOPorts = port_names_electrical,
    port_types: typings.IOPorts = port_types_electrical,
    **kwargs: Any,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    radius = radius or width
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
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
        Section(
            width=gap,
            layer=layer_port,
            offset=0,
            port_names=port_names_electrical,
            port_types=port_types_electrical,
        ),
        Section(width=width, layer=layer, offset=+gap / 2 + width / 2),
        Section(width=width, layer=layer, offset=-gap / 2 - width / 2),
    ]
    return CrossSection(sections=tuple(sections), radius=radius or 2 * width + gap)


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
        Section(
            width=width,
            layer=layer,
            offset=0,
            port_names=port_names_electrical,
            port_types=port_types_electrical,
        ),
        Section(width=width, layer=layer, offset=-gap - width),
        Section(width=width, layer=layer, offset=+gap + width),
    ]
    return CrossSection(sections=tuple(sections), radius=radius or 3 * width + 2 * gap)


@xsection
def metal_routing(
    width: float = 10,
    layer: typings.LayerSpec = "M3",
    radius: float | None = None,
    port_names: typings.IOPorts = port_names_electrical,
    port_types: typings.IOPorts = port_types_electrical,
    **kwargs: Any,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    radius = radius or width

    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
        **kwargs,
    )


@xsection
def heater_metal(
    width: float = 2.5,
    layer: typings.LayerSpec = "HEATER",
    radius: float | None = None,
    port_names: typings.IOPorts = port_names_electrical,
    port_types: typings.IOPorts = port_types_electrical,
    **kwargs: Any,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    radius = radius or width
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
        **kwargs,
    )


@xsection
def npp(
    width: float = 0.5,
    layer: typings.LayerSpec = "NPP",
    radius: float | None = None,
    port_names: typings.IOPorts = port_names_electrical,
    port_types: typings.IOPorts = port_types_electrical,
    **kwargs: Any,
) -> CrossSection:
    """Return Doped NPP cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
        **kwargs,
    )
