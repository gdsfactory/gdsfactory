"""P-N junction and doped waveguide cross-section factories."""

from __future__ import annotations

from typing import Any

from gdsfactory import typings
from gdsfactory.cross_section.base import (
    CrossSection,
    Sections,
    SectionSpec,
    cladding_layers_optical,
    cladding_offsets_optical,
)
from gdsfactory.cross_section.presets import strip
from gdsfactory.cross_section.utils import cross_section, xsection


@xsection
def pin(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    layer_slab: typings.LayerSpec = "SLAB90",
    layers_via_stack1: typings.LayerSpecs = ("PPP",),
    layers_via_stack2: typings.LayerSpecs = ("NPP",),
    bbox_offsets_via_stack1: tuple[float, ...] = (0, -0.2),
    bbox_offsets_via_stack2: tuple[float, ...] = (0, -0.2),
    via_stack_width: float = 9.0,
    via_stack_gap: float = 0.55,
    slab_gap: float = -0.2,
    layer_via: typings.LayerSpec | None = None,
    via_width: float = 1,
    via_offsets: tuple[float, ...] | None = None,
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Rib PIN doped cross_section.

    Args:
        width: ridge width.
        layer: ridge layer.
        layer_slab: slab layer.
        layers_via_stack1: list of bot layer.
        layers_via_stack2: list of top layer.
        bbox_offsets_via_stack1: for bot.
        bbox_offsets_via_stack2: for top.
        via_stack_width: in um.
        via_stack_gap: offset from via_stack to ridge edge.
        slab_gap: extra slab gap (negative: via_stack goes beyond slab).
        layer_via: for via.
        via_width: in um.
        via_offsets: in um.
        sections: cross_section sections.
        kwargs: cross_section settings.


    https://doi.org/10.1364/OE.26.029983

    ```text
                                      layer
                                |<----width--->|
                                 _______________ via_stack_gap           slab_gap
                                |              |<----------->|             <-->
        ___ ____________________|              |__________________________|___
       |   |         |                                       |            |   |
       |   |    P++  |         undoped silicon               |     N++    |   |
       |___|_________|_______________________________________|____________|___|
                                                              <----------->
                                                              via_stack_width
       <---------------------------------------------------------------------->
                                   slab_width
    ```

    Example:
        ```python
        import gdsfactory as gf

        xs = gf.cross_section.pin(width=0.5, via_stack_gap=1, via_stack_width=1)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
        ```
    """
    section_list: list[SectionSpec] = list(sections or [])
    slab_width = width + 2 * via_stack_gap + 2 * via_stack_width - 2 * slab_gap
    via_stack_offset = width / 2 + via_stack_gap + via_stack_width / 2

    section_list += [(layer_slab, -(slab_width / 2), slab_width / 2)]
    section_list += [
        (
            layer,
            via_stack_offset - (via_stack_width + 2 * cladding_offset) / 2,
            via_stack_offset + (via_stack_width + 2 * cladding_offset) / 2,
        )
        for layer, cladding_offset in zip(
            layers_via_stack1, bbox_offsets_via_stack1, strict=False
        )
    ]
    section_list += [
        (
            layer,
            -via_stack_offset - (via_stack_width + 2 * cladding_offset) / 2,
            -via_stack_offset + (via_stack_width + 2 * cladding_offset) / 2,
        )
        for layer, cladding_offset in zip(
            layers_via_stack2, bbox_offsets_via_stack2, strict=False
        )
    ]
    if layer_via and via_width and via_offsets:
        section_list += [
            (layer_via, offset - via_width / 2, offset + via_width / 2)
            for offset in via_offsets
        ]

    return strip(
        width=width,
        layer=layer,
        sections=tuple(section_list),
        **kwargs,
    )


@xsection
def pn(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    layer_slab: typings.LayerSpec = "SLAB90",
    gap_low_doping: float = 0.0,
    gap_medium_doping: float = 0.5,
    gap_high_doping: float = 1.0,
    offset_low_doping: float = 0.0,
    width_doping: float = 8.0,
    width_slab: float = 7.0,
    layer_p: typings.LayerSpec | None = "P",
    layer_pp: typings.LayerSpec | None = "PP",
    layer_ppp: typings.LayerSpec | None = "PPP",
    layer_n: typings.LayerSpec | None = "N",
    layer_np: typings.LayerSpec | None = "NP",
    layer_npp: typings.LayerSpec | None = "NPP",
    layer_via: typings.LayerSpec | None = None,
    width_via: float = 1.0,
    layer_metal: typings.LayerSpec | None = None,
    width_metal: float = 1.0,
    sections: Sections | None = None,
    cladding_layers: typings.LayerSpecs | None = None,
    cladding_offsets: typings.Floats | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Rib PN doped cross_section.

    Args:
        width: width of the ridge in um.
        layer: ridge layer.
        layer_slab: slab layer.
        gap_low_doping: from waveguide center to low doping. Only used for PIN.
        gap_medium_doping: from waveguide center to medium doping. None removes it.
        gap_high_doping: from center to high doping. None removes it.
        offset_low_doping: from center to junction center.
        width_doping: in um.
        width_slab: in um.
        layer_p: p doping layer.
        layer_pp: p+ doping layer.
        layer_ppp: p++ doping layer.
        layer_n: n doping layer.
        layer_np: n+ doping layer.
        layer_npp: n++ doping layer.
        layer_via: via layer.
        width_via: via width in um.
        layer_metal: metal layer.
        width_metal: metal width in um.
        sections: optional list of sections.
        cladding_layers: optional list of cladding layers.
        cladding_offsets: optional list of cladding offsets.
        kwargs: cross_section settings.

                              offset_low_doping
                                <------>
                               |       |
                              wg     junction
                            center   center
                               |       |
                 ______________|_______|______
                 |             |       |     |
        _________|             |       |     |_________________|
              P                |       |               N       |
          width_p              |       |            width_n    |
        <----------------------------->|<--------------------->|
                               |               |       N+      |
                               |               |    width_n    |
                               |               |<------------->|
                               |<------------->|
                               gap_medium_doping

    Example:
        ```python
        import gdsfactory as gf

        xs = gf.cross_section.pn(width=0.5, gap_low_doping=0, width_doping=2.)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
        ```
    """
    slab = (layer_slab, -(width_slab / 2), width_slab / 2)

    section_list: list[SectionSpec] = list(sections or [])
    section_list += [slab]
    base_offset_low_doping = width_doping / 2 + gap_low_doping / 4
    width_low_doping = width_doping - gap_low_doping / 2

    if layer_n:
        n = (
            layer_n,
            base_offset_low_doping
            - offset_low_doping / 2
            - (width_low_doping + offset_low_doping) / 2,
            base_offset_low_doping
            - offset_low_doping / 2
            + (width_low_doping + offset_low_doping) / 2,
        )
        section_list.append(n)
    if layer_p:
        p = (
            layer_p,
            -base_offset_low_doping
            - offset_low_doping / 2
            - (width_low_doping - offset_low_doping) / 2,
            -base_offset_low_doping
            - offset_low_doping / 2
            + (width_low_doping - offset_low_doping) / 2,
        )
        section_list.append(p)

    width_medium_doping = width_doping - gap_medium_doping
    offset_medium_doping = width_medium_doping / 2 + gap_medium_doping

    if layer_np is not None:
        np = (
            layer_np,
            offset_medium_doping - width_medium_doping / 2,
            offset_medium_doping + width_medium_doping / 2,
        )
        section_list.append(np)
    if layer_pp is not None:
        pp = (
            layer_pp,
            -offset_medium_doping - width_medium_doping / 2,
            -offset_medium_doping + width_medium_doping / 2,
        )
        section_list.append(pp)

    width_high_doping = width_doping - gap_high_doping
    offset_high_doping = width_high_doping / 2 + gap_high_doping
    if layer_npp is not None:
        npp = (
            layer_npp,
            offset_high_doping - width_high_doping / 2,
            offset_high_doping + width_high_doping / 2,
        )
        section_list.append(npp)
    if layer_ppp is not None:
        ppp = (
            layer_ppp,
            -offset_high_doping - width_high_doping / 2,
            -offset_high_doping + width_high_doping / 2,
        )
        section_list.append(ppp)

    if layer_via is not None:
        offset = width_high_doping + gap_high_doping - width_via / 2
        via_top = (layer_via, offset - width_via / 2, offset + width_via / 2)
        via_bot = (layer_via, -offset - width_via / 2, -offset + width_via / 2)
        section_list.append(via_top)
        section_list.append(via_bot)

    if layer_metal is not None:
        offset = width_high_doping + gap_high_doping - width_metal / 2
        metal_top = (
            layer_metal,
            offset - width_via / 2,
            offset + width_via / 2,
        )
        metal_bot = (
            layer_metal,
            -offset - width_via / 2,
            -offset + width_via / 2,
        )
        section_list.append(metal_top)
        section_list.append(metal_bot)

    return cross_section(
        width=width,
        offset=0,
        layer=layer,
        sections=tuple(section_list),
        cladding_offsets=cladding_offsets,
        cladding_layers=cladding_layers,
        **kwargs,
    )


@xsection
def pn_with_trenches(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    layer_trench: typings.LayerSpec = "DEEP_ETCH",
    gap_low_doping: float = 0.0,
    gap_medium_doping: float | None = 0.5,
    gap_high_doping: float | None = 1.0,
    offset_low_doping: float = 0.0,
    width_doping: float = 8.0,
    slab_offset: float | None = 0.3,
    width_slab: float | None = None,
    width_trench: float = 2.0,
    layer_p: typings.LayerSpec | None = "P",
    layer_pp: typings.LayerSpec | None = "PP",
    layer_ppp: typings.LayerSpec | None = "PPP",
    layer_n: typings.LayerSpec | None = "N",
    layer_np: typings.LayerSpec | None = "NP",
    layer_npp: typings.LayerSpec | None = "NPP",
    layer_via: typings.LayerSpec | None = None,
    width_via: float = 1.0,
    layer_metal: typings.LayerSpec | None = None,
    width_metal: float = 1.0,
    cladding_layers: typings.Layers | None = cladding_layers_optical,
    cladding_offsets: typings.Floats | None = cladding_offsets_optical,
    wg_marking_layer: typings.LayerSpec | None = None,
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Rib PN doped cross_section.

    Args:
        width: width of the ridge in um.
        layer: ridge layer. None adds only ridge.
        layer_trench: layer to etch trenches.
        gap_low_doping: from waveguide center to low doping. Only used for PIN.
        gap_medium_doping: from waveguide center to medium doping. None removes it.
        gap_high_doping: from center to high doping. None removes it.
        offset_low_doping: from center to junction center.
        width_doping: in um.
        slab_offset: from the edge of the trench to the edge of the slab.
        width_slab: in um.
        width_trench: in um.
        layer_p: p doping layer.
        layer_pp: p+ doping layer.
        layer_ppp: p++ doping layer.
        layer_n: n doping layer.
        layer_np: n+ doping layer.
        layer_npp: n++ doping layer.
        layer_via: via layer.
        width_via: via width in um.
        layer_metal: metal layer.
        width_metal: metal width in um.
        cladding_layers: optional list of cladding layers.
        cladding_offsets: optional list of cladding offsets.
        wg_marking_layer: layer to draw over the actual waveguide.
        sections: optional list of sections.
        kwargs: cross_section settings.

    ```text
                                   offset_low_doping
                                     <------>
                                    |       |
                                   wg     junction
                                 center   center             slab_offset
                                    |       |               <------>
        _____         ______________|_______ ______         ________
             |        |             |       |     |         |       |
             |________|             |             |_________|       |
                   P                |       |               N       |
               width_p              |                    width_n    |
          <-------------------------------->|<--------------------->|
             <------->              |               |       N+      |
            width_trench            |               |    width_n    |
                                    |               |<------------->|
                                    |<------------->|
                                    gap_medium_doping
       <------------------------------------------------------------>
                                width_slab
    ```

    Example:
        ```python
        import gdsfactory as gf

        xs = gf.cross_section.pn_with_trenches(width=0.5, gap_low_doping=0, width_doping=2.)
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
    section_list: list[SectionSpec] = list(sections or [])
    assert width_slab is not None
    section_list += [(layer, -(width_slab / 2), width_slab / 2)]
    section_list += [
        (layer_trench, offset - width_trench / 2, offset + width_trench / 2)
        for offset in [+trench_offset, -trench_offset]
    ]

    if wg_marking_layer is not None:
        section_list += [(wg_marking_layer, -(width / 2), width / 2)]

    base_offset_low_doping = width_doping / 2 + gap_low_doping / 4
    width_low_doping = width_doping - gap_low_doping / 2

    if layer_n:
        n = (
            layer_n,
            base_offset_low_doping
            - offset_low_doping / 2
            - (width_low_doping + offset_low_doping) / 2,
            base_offset_low_doping
            - offset_low_doping / 2
            + (width_low_doping + offset_low_doping) / 2,
        )
        section_list.append(n)
    if layer_p:
        p = (
            layer_p,
            -base_offset_low_doping
            - offset_low_doping / 2
            - (width_low_doping - offset_low_doping) / 2,
            -base_offset_low_doping
            - offset_low_doping / 2
            + (width_low_doping - offset_low_doping) / 2,
        )
        section_list.append(p)

    if gap_medium_doping is not None:
        width_medium_doping = width_doping - gap_medium_doping
        offset_medium_doping = width_medium_doping / 2 + gap_medium_doping

        if layer_np:
            np = (
                layer_np,
                offset_medium_doping - width_medium_doping / 2,
                offset_medium_doping + width_medium_doping / 2,
            )
            section_list.append(np)
        if layer_pp:
            pp = (
                layer_pp,
                -offset_medium_doping - width_medium_doping / 2,
                -offset_medium_doping + width_medium_doping / 2,
            )
            section_list.append(pp)

    width_high_doping: float | None = None
    if gap_high_doping is not None:
        width_high_doping = width_doping - gap_high_doping
        offset_high_doping = width_high_doping / 2 + gap_high_doping
        if layer_npp:
            npp = (
                layer_npp,
                offset_high_doping - width_high_doping / 2,
                offset_high_doping + width_high_doping / 2,
            )
            section_list.append(npp)
        if layer_ppp:
            ppp = (
                layer_ppp,
                -offset_high_doping - width_high_doping / 2,
                -offset_high_doping + width_high_doping / 2,
            )
            section_list.append(ppp)

    if (
        layer_via is not None
        and gap_high_doping is not None
        and width_high_doping is not None
    ):
        offset = width_high_doping + gap_high_doping - width_via / 2
        via_top = (layer_via, offset - width_via / 2, offset + width_via / 2)
        via_bot = (layer_via, -offset - width_via / 2, -offset + width_via / 2)
        section_list.append(via_top)
        section_list.append(via_bot)

    if (
        layer_metal is not None
        and width_high_doping is not None
        and gap_high_doping is not None
    ):
        offset = width_high_doping + gap_high_doping - width_metal / 2
        metal_top = (
            layer_metal,
            offset - width_via / 2,
            offset + width_via / 2,
        )
        metal_bot = (
            layer_metal,
            -offset - width_via / 2,
            -offset + width_via / 2,
        )
        section_list.append(metal_top)
        section_list.append(metal_bot)

    return cross_section(
        width=width,
        offset=0,
        layer=layer,
        sections=tuple(section_list),
        cladding_offsets=cladding_offsets,
        cladding_layers=cladding_layers,
        **kwargs,
    )


@xsection
def pn_with_trenches_asymmetric(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    layer_trench: typings.LayerSpec = "DEEP_ETCH",
    gap_low_doping: float | tuple[float, float] = (0.0, 0.0),
    gap_medium_doping: float | tuple[float, float] | None = (0.5, 0.2),
    gap_high_doping: float | tuple[float, float] | None = (1.0, 0.8),
    width_doping: float = 8.0,
    slab_offset: float | None = 0.3,
    width_slab: float | None = None,
    width_trench: float = 2.0,
    layer_p: typings.LayerSpec | None = "P",
    layer_pp: typings.LayerSpec | None = "PP",
    layer_ppp: typings.LayerSpec | None = "PPP",
    layer_n: typings.LayerSpec | None = "N",
    layer_np: typings.LayerSpec | None = "NP",
    layer_npp: typings.LayerSpec | None = "NPP",
    layer_via: typings.LayerSpec | None = None,
    width_via: float = 1.0,
    layer_metal: typings.LayerSpec | None = None,
    width_metal: float = 1.0,
    cladding_layers: typings.Layers | None = cladding_layers_optical,
    cladding_offsets: typings.Floats | None = cladding_offsets_optical,
    wg_marking_layer: typings.LayerSpec | None = None,
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Rib PN doped cross_section with asymmetric dimensions left and right.

    Args:
        width: width of the ridge in um.
        layer: ridge layer. None adds only ridge.
        layer_trench: layer to etch trenches.
        gap_low_doping: from waveguide center to low doping. Only used for PIN. \
                If a list, it considers the first element is [p_side, n_side]. If a number, \
                it assumes the same for both sides.
        gap_medium_doping: from waveguide center to medium doping. None removes it. \
                If a list, it considers the first element is [p_side, n_side]. \
                If a number, it assumes the same for both sides.
        gap_high_doping: from center to high doping. None removes it. \
                If a list, it considers the first element is [p_side, n_side].\
                If a number, it assumes the same for both sides.
        width_doping: in um.
        slab_offset: from the edge of the trench to the edge of the slab.
        width_slab: in um.
        width_trench: in um.
        layer_p: p doping layer.
        layer_pp: p+ doping layer.
        layer_ppp: p++ doping layer.
        layer_n: n doping layer.
        layer_np: n+ doping layer.
        layer_npp: n++ doping layer.
        layer_via: via layer.
        width_via: via width in um.
        layer_metal: metal layer.
        width_metal: metal width in um.
        cladding_layers: optional list of cladding layers.
        cladding_offsets: optional list of cladding offsets.
        wg_marking_layer: layer to draw over the actual waveguide.
        sections: optional list of sections.
        kwargs: cross_section settings.

    ```text
                                   gap_low_doping[1]
                                     <------>
                                    |       |
                                   wg     junction
                                 center   center           slab_offset
                                    |       |               <------>
        _____         ______________|_______ ______         ________
             |        |             |       |     |         |       |
             |________|             |             |_________|       |
                   P                |       |               N       |
               width_p              |                    width_n    |
          <-------------------------------->|<--------------------->|
             <------->              |               |       N+      |
            width_trench            |               |    width_n    |
                                    |               |<------------->|
                                    |<------------->|
                                    gap_medium_doping[1]
       <------------------------------------------------------------>
                                width_slab
    ```

    Example:
        ```python
        import gdsfactory as gf

        xs = gf.cross_section.pn_with_trenches_assymmetric(width=0.5, gap_low_doping=0, width_doping=2.)
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

    # Trenches
    trench_offset = width / 2 + width_trench / 2
    section_list: list[SectionSpec] = list(sections or [])
    assert width_slab is not None
    section_list += [(layer, -(width_slab / 2), width_slab / 2)]
    section_list += [
        (layer_trench, offset - width_trench / 2, offset + width_trench / 2)
        for offset in [+trench_offset, -trench_offset]
    ]

    if wg_marking_layer is not None:
        section_list += [(wg_marking_layer, -(width / 2), width / 2)]

    # Low doping

    if not isinstance(gap_low_doping, list | tuple):
        gap_low_doping_list = [gap_low_doping] * 2
    else:
        gap_low_doping_list = list(gap_low_doping)

    if layer_n:
        width_low_doping_n = width_doping - gap_low_doping_list[1]
        n = (
            layer_n,
            width_low_doping_n / 2 + gap_low_doping_list[1] - width_low_doping_n / 2,
            width_low_doping_n / 2 + gap_low_doping_list[1] + width_low_doping_n / 2,
        )
        section_list.append(n)
    if layer_p:
        width_low_doping_p = width_doping - gap_low_doping_list[0]
        p = (
            layer_p,
            -(width_low_doping_p / 2 + gap_low_doping_list[0]) - width_low_doping_p / 2,
            -(width_low_doping_p / 2 + gap_low_doping_list[0]) + width_low_doping_p / 2,
        )
        section_list.append(p)

    if gap_medium_doping is not None:
        if not isinstance(gap_medium_doping, list | tuple):
            gap_medium_doping_list = [gap_medium_doping] * 2
        else:
            gap_medium_doping_list = list(gap_medium_doping)

        if layer_np:
            width_np = width_doping - gap_medium_doping_list[1]
            np = (
                layer_np,
                width_np / 2 + gap_medium_doping_list[1] - width_np / 2,
                width_np / 2 + gap_medium_doping_list[1] + width_np / 2,
            )
            section_list.append(np)
        if layer_pp:
            width_pp = width_doping - gap_medium_doping_list[0]
            pp = (
                layer_pp,
                -(width_pp / 2 + gap_medium_doping_list[0]) - width_pp / 2,
                -(width_pp / 2 + gap_medium_doping_list[0]) + width_pp / 2,
            )
            section_list.append(pp)
    gap_high_doping_list: list[float] | None = None
    width_npp: float | None = None
    width_ppp: float | None = None
    if gap_high_doping is not None:
        if not isinstance(gap_high_doping, list | tuple):
            gap_high_doping_list = [float(gap_high_doping)] * 2
        else:
            gap_high_doping_list = list(gap_high_doping)

        if layer_npp:
            width_npp = width_doping - gap_high_doping_list[1]
            npp = (
                layer_npp,
                width_npp / 2 + gap_high_doping_list[1] - width_npp / 2,
                width_npp / 2 + gap_high_doping_list[1] + width_npp / 2,
            )
            section_list.append(npp)
        if layer_ppp:
            width_ppp = width_doping - gap_high_doping_list[0]
            ppp = (
                layer_ppp,
                -(width_ppp / 2 + gap_high_doping_list[0]) - width_ppp / 2,
                -(width_ppp / 2 + gap_high_doping_list[0]) + width_ppp / 2,
            )
            section_list.append(ppp)

    if (
        layer_via is not None
        and gap_high_doping_list is not None
        and width_npp is not None
        and width_ppp is not None
    ):
        offset_top = width_npp + gap_high_doping_list[1] - width_via / 2
        offset_bot = width_ppp + gap_high_doping_list[0] - width_via / 2
        via_top = (
            layer_via,
            offset_top - width_via / 2,
            offset_top + width_via / 2,
        )
        via_bot = (
            layer_via,
            -offset_bot - width_via / 2,
            -offset_bot + width_via / 2,
        )
        section_list.append(via_top)
        section_list.append(via_bot)

    if (
        layer_metal is not None
        and gap_high_doping_list is not None
        and width_npp is not None
        and width_ppp is not None
    ):
        offset_top = width_npp + gap_high_doping_list[1] - width_metal / 2
        offset_bot = width_ppp + gap_high_doping_list[0] - width_metal / 2
        metal_top = (
            layer_metal,
            offset_top - width_via / 2,
            offset_top + width_via / 2,
        )
        metal_bot = (
            layer_metal,
            -offset_bot - width_via / 2,
            -offset_bot + width_via / 2,
        )
        section_list.append(metal_top)
        section_list.append(metal_bot)

    return cross_section(
        width=width,
        offset=0,
        layer=layer,
        sections=tuple(section_list),
        cladding_offsets=cladding_offsets,
        cladding_layers=cladding_layers,
        **kwargs,
    )


@xsection
def l_wg_doped_with_trenches(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    layer_trench: typings.LayerSpec = "DEEP_ETCH",
    gap_low_doping: float = 0.0,
    gap_medium_doping: float | None = 0.5,
    gap_high_doping: float | None = 1.0,
    width_doping: float = 8.0,
    slab_offset: float | None = 0.3,
    width_slab: float | None = None,
    width_trench: float = 2.0,
    layer_low: typings.LayerSpec = "P",
    layer_mid: typings.LayerSpec = "PP",
    layer_high: typings.LayerSpec = "PPP",
    layer_via: typings.LayerSpec | None = None,
    width_via: float = 1.0,
    layer_metal: typings.LayerSpec | None = None,
    width_metal: float = 1.0,
    cladding_layers: typings.Layers | None = cladding_layers_optical,
    cladding_offsets: typings.Floats | None = cladding_offsets_optical,
    wg_marking_layer: typings.LayerSpec | None = None,
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """L waveguide PN doped cross_section.

    Args:
        width: width of the ridge in um.
        layer: ridge layer. None adds only ridge.
        layer_trench: layer to etch trenches.
        gap_low_doping: from waveguide outer edge to low doping. Only used for PIN.
        gap_medium_doping: from waveguide edge to medium doping. None removes it.
        gap_high_doping: from edge to high doping. None removes it.
        width_doping: in um.
        slab_offset: from the edge of the trench to the edge of the slab.
        width_slab: in um.
        width_trench: in um.
        layer_low: low doping layer.
        layer_mid: mid doping layer.
        layer_high: high doping layer.
        layer_via: via layer.
        width_via: via width in um.
        layer_metal: metal layer.
        width_metal: metal width in um.
        cladding_layers: optional list of cladding layers.
        cladding_offsets: optional list of cladding offsets.
        wg_marking_layer: layer to mark where the actual guiding section is.
        sections: optional list of sections.
        kwargs: cross_section settings.

    ```text
                                          gap_low_doping
                                           <------>
                                                  |
                                                  wg
                                                 edge
                                                  |
        _____                       _______ ______
             |                     |              |
             |_____________________|              |
                                                  |
                                                  |
                                    <------------>
                                           width
             <--------------------->               |
            width_trench       |                   |
                               |                   |
                               |<----------------->|
                                  gap_medium_doping
                     |<--------------------------->|
                             gap_high_doping
       <------------------------------------------->
                        width_slab
    ```

    Example:
        ```python
        import gdsfactory as gf

        xs = gf.cross_section.pn_with_trenches(width=0.5, gap_low_doping=0, width_doping=2.)
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

    trench_offset = -1 * (width / 2 + width_trench / 2)
    section_list: list[SectionSpec] = list(sections or [])
    assert width_slab is not None
    section_list.append(
        (
            layer,
            -1 * (width_slab / 2 - width / 2) - width_slab / 2,
            -1 * (width_slab / 2 - width / 2) + width_slab / 2,
        )
    )
    section_list += [
        (
            layer_trench,
            trench_offset - width_trench / 2,
            trench_offset + width_trench / 2,
        )
    ]

    if wg_marking_layer is not None:
        section_list += [(wg_marking_layer, -(width / 2), width / 2)]

    offset_low_doping = width / 2 - gap_low_doping - width_doping / 2

    low_doping = (
        layer_low,
        offset_low_doping - width_doping / 2,
        offset_low_doping + width_doping / 2,
    )

    section_list.append(low_doping)

    if gap_medium_doping is not None:
        width_medium_doping = width_doping - gap_medium_doping
        offset_medium_doping = width / 2 - gap_medium_doping - width_medium_doping / 2

        mid_doping = (
            layer_mid,
            offset_medium_doping - width_medium_doping / 2,
            offset_medium_doping + width_medium_doping / 2,
        )
        section_list.append(mid_doping)

    offset_high_doping: float | None = None
    width_high_doping: float | None = None

    if gap_high_doping is not None:
        width_high_doping = width_doping - gap_high_doping
        offset_high_doping = width / 2 - gap_high_doping - width_high_doping / 2

        high_doping = (
            layer_high,
            offset_high_doping - width_high_doping / 2,
            offset_high_doping + width_high_doping / 2,
        )

        section_list.append(high_doping)

    if (
        layer_via is not None
        and offset_high_doping is not None
        and width_high_doping is not None
    ):
        offset = offset_high_doping - width_high_doping / 2 + width_via / 2
        via = (layer_via, offset - width_via / 2, offset + width_via / 2)
        section_list.append(via)

    if (
        layer_metal is not None
        and offset_high_doping is not None
        and width_high_doping is not None
    ):
        offset = offset_high_doping - width_high_doping / 2 + width_metal / 2
        metal = (layer_metal, offset - width_via / 2, offset + width_via / 2)
        section_list.append(metal)

    return cross_section(
        width=width,
        offset=0,
        layer=layer,
        sections=tuple(section_list),
        cladding_offsets=cladding_offsets,
        cladding_layers=cladding_layers,
        **kwargs,
    )


@xsection
def pn_ge_detector_si_contacts(
    width_si: float = 6.0,
    layer_si: typings.LayerSpec = "WG",
    width_ge: float = 3.0,
    layer_ge: typings.LayerSpec = "GE",
    gap_low_doping: float = 0.6,
    gap_medium_doping: float = 0.9,
    gap_high_doping: float = 1.1,
    width_doping: float = 8.0,
    layer_p: typings.LayerSpec = "P",
    layer_pp: typings.LayerSpec = "PP",
    layer_ppp: typings.LayerSpec = "PPP",
    layer_n: typings.LayerSpec = "N",
    layer_np: typings.LayerSpec = "NP",
    layer_npp: typings.LayerSpec = "NPP",
    layer_via: typings.LayerSpec | None = None,
    width_via: float = 1.0,
    layer_metal: typings.LayerSpec | None = None,
    cladding_layers: typings.Layers | None = cladding_layers_optical,
    cladding_offsets: typings.Floats | None = cladding_offsets_optical,
    **kwargs: Any,
) -> CrossSection:
    """Linear Ge detector cross section based on a lateral p(i)n junction.

    It has silicon contacts (no contact on the Ge). The contacts need to be
    created in the component generating function (they can't be created here).

    See Chen et al., "High-Responsivity Low-Voltage 28-Gb/s Ge p-i-n Photodetector
    With Silicon Contacts", Journal of Lightwave Technology 33(4), 2015.

    Notice it is possible to have dopings going beyond the ridge waveguide. This
    is fine, and it is to account for the
    presence of the contacts. Such contacts can be subwavelength or not.

    Args:
        width_si: width of the full etch si in um.
        layer_si: si ridge layer.
        width_ge: width of the ge in um.
        layer_ge: ge layer.
        gap_low_doping: from waveguide center to low doping.
        gap_medium_doping: from waveguide center to medium doping. None removes it.
        gap_high_doping: from center to high doping. None removes it.
        width_doping: distance from waveguide center to doping edge in um.
        layer_p: p doping layer.
        layer_pp: p+ doping layer.
        layer_ppp: p++ doping layer.
        layer_n: n doping layer.
        layer_np: n+ doping layer.
        layer_npp: n++ doping layer.
        layer_via: via layer.
        width_via: via width in um.
        layer_metal: metal layer.
        cladding_layers: list of layers to extrude.
        cladding_offsets: list of offset from main strip edge.
        kwargs: cross_section settings.

                                   layer_si
                           |<------width_si---->|

                                  layer_ge
                              |<--width_ge->|
                               ______________
                              |             |
                            __|_____________|___
                           |     |       |     |
                           |     |       |     |
                    P      |     |       |     |         N                |
                 width_p   |_____|_______|_____|           width_n        |
        <----------------------->|       |<------------------------------>|
                                     |<->|
                                     gap_low_doping
                                     |         |        N+                |
                                     |         |     width_np             |
                                     |         |<------------------------>|
                                     |<------->|
                                     |     gap_medium_doping
                                     |
                                     |<---------------------------------->|
                                                width_doping

    Example:
        ```python
        import gdsfactory as gf

        xs = gf.cross_section.pn()
        p = gf.path.straight()
        c = p.extrude(xs)
        c.plot()
        ```
    """
    width_low_doping = width_doping - gap_low_doping
    offset_low_doping = width_low_doping / 2 + gap_low_doping

    s = (layer_si, -(width_si / 2), width_si / 2)
    n = (
        layer_n,
        offset_low_doping - width_low_doping / 2,
        offset_low_doping + width_low_doping / 2,
    )
    p = (
        layer_p,
        -offset_low_doping - width_low_doping / 2,
        -offset_low_doping + width_low_doping / 2,
    )

    section_list = [s, n, p]

    cladding_layers = cladding_layers or ()
    cladding_offsets = cladding_offsets or ()
    section_list += [
        (layer, -((width_si + 2 * offset) / 2), (width_si + 2 * offset) / 2)
        for layer, offset in zip(cladding_layers, cladding_offsets, strict=True)
    ]

    width_medium_doping = width_doping - gap_medium_doping
    offset_medium_doping = width_medium_doping / 2 + gap_medium_doping

    np = (
        layer_np,
        offset_medium_doping - width_medium_doping / 2,
        offset_medium_doping + width_medium_doping / 2,
    )
    pp = (
        layer_pp,
        -offset_medium_doping - width_medium_doping / 2,
        -offset_medium_doping + width_medium_doping / 2,
    )
    section_list.extend((np, pp))
    width_high_doping = width_doping - gap_high_doping
    offset_high_doping = width_high_doping / 2 + gap_high_doping
    npp = (
        layer_npp,
        offset_high_doping - width_high_doping / 2,
        offset_high_doping + width_high_doping / 2,
    )
    ppp = (
        layer_ppp,
        -offset_high_doping - width_high_doping / 2,
        -offset_high_doping + width_high_doping / 2,
    )
    section_list.extend((npp, ppp))
    if layer_via is not None:
        offset = width_high_doping / 2 + gap_high_doping
        via_top = (layer_via, offset - width_via / 2, offset + width_via / 2)
        via_bot = (layer_via, -offset - width_via / 2, -offset + width_via / 2)
        section_list.extend((via_top, via_bot))
    if layer_metal is not None:
        offset = width_high_doping / 2 + gap_high_doping
        metal_top = (
            layer_metal,
            offset - width_via / 2,
            offset + width_via / 2,
        )
        metal_bot = (
            layer_metal,
            -offset - width_via / 2,
            -offset + width_via / 2,
        )
        section_list.extend((metal_top, metal_bot))

    # Add the Ge
    s = (layer_ge, -(width_ge / 2), width_ge / 2)
    section_list.append(s)

    return cross_section(
        width=None,
        sections=tuple(section_list),
        **kwargs,
    )
