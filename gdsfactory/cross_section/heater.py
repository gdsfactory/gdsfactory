"""Heater cross-section variants."""

from __future__ import annotations

from typing import Any

from gdsfactory import typings
from gdsfactory.cross_section.base import (
    CrossSection,
    Section,
    Sections,
    port_names_electrical,
    port_types_electrical,
)
from gdsfactory.cross_section.presets import strip
from gdsfactory.cross_section.utils import xsection


@xsection
def strip_heater_metal_undercut(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    heater_width: float = 2.5,
    trench_width: float = 6.5,
    trench_gap: float = 2.0,
    layer_heater: typings.LayerSpec = "HEATER",
    layer_trench: typings.LayerSpec = "DEEPTRENCH",
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Returns strip cross_section with top metal and undercut trenches on both.

    sides.

    dimensions from https://doi.org/10.1364/OE.18.020298

    Args:
        width: waveguide width.
        layer: waveguide layer.
        heater_width: of metal heater.
        trench_width: in um.
        trench_gap: from waveguide edge to trench edge.
        layer_heater: heater layer.
        layer_trench: tench layer.
        sections: cross_section sections.
        kwargs: cross_section settings.

    .. code::

              |<-------heater_width--------->|
               ______________________________
              |                              |
              |         layer_heater         |
              |______________________________|

                   |<------width------>|
                    ____________________ trench_gap
                   |                   |<----------->|              |
                   |                   |             |   undercut   |
                   |       width       |             |              |
                   |                   |             |<------------>|
                   |___________________|             | trench_width |
                                                     |              |
                                                     |              |

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.strip_heater_metal_undercut(width=0.5, heater_width=2, trench_width=4, trench_gap=4)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    trench_offset = trench_gap + trench_width / 2 + width / 2
    section_list: list[Section] = list(sections or [])
    section_list += [
        Section(
            layer=layer_heater,
            width=heater_width,
            port_names=port_names_electrical,
            port_types=port_types_electrical,
        ),
        Section(layer=layer_trench, width=trench_width, offset=+trench_offset),
        Section(layer=layer_trench, width=trench_width, offset=-trench_offset),
    ]

    return strip(
        width=width,
        layer=layer,
        sections=tuple(section_list),
        **kwargs,
    )


@xsection
def strip_heater_metal(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    heater_width: float = 2.5,
    layer_heater: typings.LayerSpec = "HEATER",
    sections: Sections | None = None,
    insets: tuple[float, float] | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Returns strip cross_section with top heater metal.

    dimensions from https://doi.org/10.1364/OE.18.020298

    Args:
        width: waveguide width (um).
        layer: waveguide layer.
        heater_width: of metal heater.
        layer_heater: for the metal.
        sections: cross_section sections.
        insets: for the heater.
        kwargs: cross_section settings.

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.strip_heater_metal(width=0.5, heater_width=2)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    section_list: list[Section] = list(sections or [])
    section_list += [
        Section(
            layer=layer_heater,
            width=heater_width,
            port_names=port_names_electrical,
            port_types=port_types_electrical,
            insets=insets,
        )
    ]

    return strip(
        width=width,
        layer=layer,
        sections=tuple(section_list),
        **kwargs,
    )


@xsection
def strip_heater_doped(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    heater_width: float = 2.0,
    heater_gap: float = 0.8,
    layers_heater: typings.LayerSpecs = ("WG", "NPP"),
    bbox_offsets_heater: tuple[float, ...] = (0, 0.1),
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Returns strip cross_section with N++ doped heaters on both sides.

    Args:
        width: in um.
        layer: waveguide spec.
        heater_width: in um.
        heater_gap: in um.
        layers_heater: for doped heater.
        bbox_offsets_heater: for each layers_heater.
        sections: cross_section sections.
        kwargs: cross_section settings.

    .. code::

                                  |<------width------>|
          ____________             ___________________               ______________
         |            |           |     undoped Si    |             |              |
         |layer_heater|           |  intrinsic region |<----------->| layer_heater |
         |____________|           |___________________|             |______________|
                                                                     <------------>
                                                        heater_gap     heater_width

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.strip_heater_doped(width=0.5, heater_width=2, heater_gap=0.5)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    heater_offset = width / 2 + heater_gap + heater_width / 2

    section_list: list[Section] = list(sections or [])
    section_list += [
        Section(
            layer=layer,
            width=heater_width + 2 * cladding_offset,
            offset=+heater_offset,
            name=f"heater_upper_{layer}",
        )
        for layer, cladding_offset in zip(
            layers_heater, bbox_offsets_heater, strict=False
        )
    ]

    section_list += [
        Section(
            layer=layer,
            width=heater_width + 2 * cladding_offset,
            offset=-heater_offset,
            name=f"heater_lower_{layer}",
        )
        for layer, cladding_offset in zip(
            layers_heater, bbox_offsets_heater, strict=False
        )
    ]

    return strip(
        width=width,
        layer=layer,
        sections=tuple(section_list),
        **kwargs,
    )


@xsection
def rib_heater_doped(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    heater_width: float = 2.0,
    heater_gap: float = 0.8,
    layer_heater: typings.LayerSpec = "NPP",
    layer_slab: typings.LayerSpec = "SLAB90",
    slab_gap: float = 0.2,
    with_top_heater: bool = True,
    with_bot_heater: bool = True,
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Returns rib cross_section with N++ doped heaters on both sides.

    dimensions from https://doi.org/10.1364/OE.27.010456

    .. code::

                                    |<------width------>|
                                     ____________________  heater_gap           slab_gap
                                    |                   |<----------->|             <-->
         ___ _______________________|                   |__________________________|___
        |   |            |                undoped Si                  |            |   |
        |   |layer_heater|                intrinsic region            |layer_heater|   |
        |___|____________|____________________________________________|____________|___|
                                                                       <---------->
                                                                        heater_width
        <------------------------------------------------------------------------------>
                                        slab_width

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.rib_heater_doped(width=0.5, heater_width=2, heater_gap=0.5, layer_heater='NPP')
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    heater_offset = width / 2 + heater_gap + heater_width / 2

    if with_bot_heater and with_top_heater:
        slab_width = width + 2 * heater_gap + 2 * heater_width + 2 * slab_gap
        slab_offset = 0.0
    elif with_top_heater:
        slab_width = width + heater_gap + heater_width + slab_gap
        slab_offset = -slab_width / 2
    elif with_bot_heater:
        slab_width = width + heater_gap + heater_width + slab_gap
        slab_offset = +slab_width / 2
    else:
        raise ValueError("At least one heater must be True")

    section_list: list[Section] = list(sections or [])
    section_list += [
        Section(width=slab_width, layer=layer_slab, offset=slab_offset, name="slab")
    ]

    if with_bot_heater:
        section_list += [
            Section(
                layer=layer_heater,
                width=heater_width,
                offset=+heater_offset,
                name="heater_upper",
            )
        ]
    if with_top_heater:
        section_list += [
            Section(
                layer=layer_heater,
                width=heater_width,
                offset=-heater_offset,
                name="heater_lower",
            )
        ]
    return strip(
        width=width,
        layer=layer,
        sections=tuple(section_list),
        **kwargs,
    )


@xsection
def rib_heater_doped_via_stack(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    heater_width: float = 1.0,
    heater_gap: float = 0.8,
    layer_slab: typings.LayerSpec = "SLAB90",
    layer_heater: typings.LayerSpec = "NPP",
    via_stack_width: float = 2.0,
    via_stack_gap: float = 0.8,
    layers_via_stack: typings.LayerSpecs = ("NPP", "VIAC"),
    bbox_offsets_via_stack: tuple[float, ...] = (0, -0.2),
    slab_gap: float = 0.2,
    slab_offset: float = 0,
    with_top_heater: bool = True,
    with_bot_heater: bool = True,
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Returns rib cross_section with N++ doped heaters on both sides.

    dimensions from https://doi.org/10.1364/OE.27.010456

    Args:
        width: in um.
        layer: for main waveguide section.
        heater_width: in um.
        heater_gap: in um.
        layer_slab: for pedestal.
        layer_heater: for doped heater.
        via_stack_width: for the contact.
        via_stack_gap: in um.
        layers_via_stack: for the contact.
        bbox_offsets_via_stack: for the contact.
        slab_gap: from heater edge.
        slab_offset: over the center of the slab.
        with_top_heater: adds top/left heater.
        with_bot_heater: adds bottom/right heater.
        sections: list of sections to add to the cross_section.
        kwargs: cross_section settings.

    .. code::

                                   |<----width------>|
       slab_gap                     __________________ via_stack_gap     via_stack width
       <-->                        |                 |<------------>|<--------------->
                                   |                 | heater_gap |
                                   |                 |<---------->|
        ___ _______________________|                 |___________________________ ____
       |   |            |              undoped Si                 |              |    |
       |   |layer_heater|              intrinsic region           |layer_heater  |    |
       |___|____________|_________________________________________|______________|____|
                                                                   <------------>
                                                                    heater_width
       <------------------------------------------------------------------------------>
                                       slab_width

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.rib_heater_doped_via_stack(width=0.5, heater_width=2, heater_gap=0.5, layer_heater='NPP')
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    if with_bot_heater and with_top_heater:
        slab_width = width + 2 * heater_gap + 2 * heater_width + 2 * slab_gap
    elif with_top_heater:
        slab_width = width + heater_gap + heater_width + slab_gap
        slab_offset -= slab_width / 2
    elif with_bot_heater:
        slab_width = width + heater_gap + heater_width + slab_gap
        slab_offset += slab_width / 2
    else:
        raise ValueError("At least one heater must be True")

    heater_offset = width / 2 + heater_gap + heater_width / 2
    via_stack_offset = width / 2 + via_stack_gap + via_stack_width / 2
    section_list: list[Section] = list(sections or [])
    section_list += [
        Section(width=slab_width, layer=layer_slab, offset=slab_offset, name="slab"),
    ]
    if with_bot_heater:
        section_list += [
            Section(
                layer=layer_heater,
                width=heater_width,
                offset=+heater_offset,
            )
        ]

    if with_top_heater:
        section_list += [
            Section(
                layer=layer_heater,
                width=heater_width,
                offset=-heater_offset,
            )
        ]

    if with_bot_heater:
        section_list += [
            Section(
                layer=layer,
                width=heater_width + 2 * cladding_offset,
                offset=+via_stack_offset,
            )
            for layer, cladding_offset in zip(
                layers_via_stack, bbox_offsets_via_stack, strict=False
            )
        ]

    if with_top_heater:
        section_list += [
            Section(
                layer=layer,
                width=heater_width + 2 * cladding_offset,
                offset=-via_stack_offset,
            )
            for layer, cladding_offset in zip(
                layers_via_stack, bbox_offsets_via_stack, strict=False
            )
        ]

    return strip(
        sections=tuple(section_list),
        width=width,
        layer=layer,
        **kwargs,
    )
