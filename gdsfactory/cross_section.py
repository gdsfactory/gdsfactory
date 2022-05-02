"""You can define a path as list of points.
To create a component you need to extrude the path with a cross-section.
"""
import inspect
import sys
from collections.abc import Iterable
from functools import partial
from inspect import getmembers
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pydantic
from pydantic import BaseModel, Field

from gdsfactory.tech import TECH, Section

LAYER = TECH.layer
Layer = Tuple[int, int]
Layers = Tuple[Layer, ...]
Floats = Tuple[float, ...]


class CrossSection(BaseModel):
    """Extend phidl.device_layout.CrossSection with port_types.

    Args:
        layer: main Section layer.
        width: (um) or function that is parameterized from 0 to 1.
            the width at t==0 is the width at the beginning of the Path.
            the width at t==1 is the width at the end.
        offset: center offset (um) or function parameterized function from 0 to 1.
             the offset at t==0 is the offset at the beginning of the Path.
             the offset at t==1 is the offset at the end.
        radius: main Section bend radius (um).
        layer_bbox: optional bounding box layer for device recognition. (68, 0)
        width_wide: wide waveguides width (um) for low loss routing.
        auto_widen: taper to wide waveguides for low loss routing.
        auto_widen_minimum_length: minimum straight length for auto_widen.
        taper_length: taper_length for auto_widen.
        bbox_layers: list of layers for rectangular bounding box.
        bbox_offsets: list of bounding box offsets.
        sections: list of Sections(width, offset, layer, ports).
        port_names: for input and output ('o1', 'o2').
        port_types: for input and output: electrical, optical, vertical_te ...
        min_length: defaults to 1nm = 10e-3um for routing.
        start_straight_length: straight length at the beginning of the route.
        end_straight_length: end length at the beginning of the route.
        snap_to_grid: can snap points to grid when extruding the path.
        aliases: dict of cross_section aliases.
        decorator: function when extruding component.
        info: dict with extra settings or useful information.
        name: cross_section name.
    """

    layer: Layer
    width: Union[float, Callable]
    offset: Union[float, Callable] = 0
    radius: Optional[float]
    layer_bbox: Optional[Tuple[int, int]] = None
    width_wide: Optional[float] = None
    auto_widen: bool = False
    auto_widen_minimum_length: float = 200.0
    taper_length: float = 10.0
    bbox_layers: List[Layer] = Field(default_factory=list)
    bbox_offsets: List[float] = Field(default_factory=list)
    sections: List[Section] = Field(default_factory=list)
    port_names: Tuple[str, str] = ("o1", "o2")
    port_types: Tuple[str, str] = ("optical", "optical")
    min_length: float = 10e-3
    start_straight_length: float = 10e-3
    end_straight_length: float = 10e-3
    snap_to_grid: Optional[float] = None
    decorator: Optional[Callable] = None
    info: Dict[str, Any] = Field(default_factory=dict)
    name: Optional[str] = None

    class Config:
        frozen = True
        extra = "forbid"

    def get_copy(self, width: Optional[float] = None):
        if not width:
            return self.copy()
        settings = dict(self)
        settings.update(width=width)
        return CrossSection(**settings)

    @property
    def aliases(self) -> Dict[str, Section]:
        s = dict(
            _default=Section(
                width=self.width,
                offset=self.offset,
                layer=self.layer,
                port_names=self.port_names,
                port_types=self.port_types,
                name="_default",
            )
        )
        sections = self.sections or []
        for section in sections:
            if section.name:
                s[section.name] = section
        return s


class Transition(CrossSection):
    cross_section1: CrossSection
    cross_section2: CrossSection
    width_type: str = "sine"
    sections: List[Section]
    layer: Optional[Layer] = None
    width: Optional[Union[float, Callable]] = None


@pydantic.validate_arguments
def cross_section(
    width: Union[Callable, float] = 0.5,
    offset: Union[float, Callable] = 0,
    layer: Tuple[int, int] = (1, 0),
    width_wide: Optional[float] = None,
    auto_widen: bool = False,
    auto_widen_minimum_length: float = 200.0,
    taper_length: float = 10.0,
    radius: Optional[float] = 10.0,
    sections: Optional[Tuple[Section, ...]] = None,
    port_names: Tuple[str, str] = ("o1", "o2"),
    port_types: Tuple[str, str] = ("optical", "optical"),
    min_length: float = 10e-3,
    start_straight_length: float = 10e-3,
    end_straight_length: float = 10e-3,
    snap_to_grid: Optional[float] = None,
    bbox_layers: Optional[List[Layer]] = None,
    bbox_offsets: Optional[List[float]] = None,
    info: Optional[Dict[str, Any]] = None,
    decorator: Optional[Callable] = None,
) -> CrossSection:
    """Return CrossSection.

    Args:
        width: (um) or function that is parameterized from 0 to 1.
            the width at t==0 is the width at the beginning of the Path.
            the width at t==1 is the width at the end.
        offset: center offset (um) or function parameterized function from 0 to 1.
             the offset at t==0 is the offset at the beginning of the Path.
             the offset at t==1 is the offset at the end.
        layer: main section layer.
        width_wide: wide waveguides width (um) for low loss routing.
        auto_widen: taper to wide waveguides for low loss routing.
        auto_widen_minimum_length: minimum straight length for auto_widen.
        taper_length: taper_length for auto_widen.
        radius: bend radius (um)..
        sections: list of Sections(width, offset, layer, ports).
        port_names: for input and output ('o1', 'o2').
        port_types: for input and output: electrical, optical, vertical_te ...
        min_length: defaults to 1nm = 10e-3um for routing.
        start_straight_length: straight length at the beginning of the route.
        end_straight_length: end length at the beginning of the route.
        snap_to_grid: can snap points to grid when extruding the path.
        bbox_layers: list of layers for rectangular bounding box.
        bbox_offsets: list of bounding box offsets.
        info: settings info.
        decorator: funcion to run when converting path to component.
    """

    return CrossSection(
        width=width,
        offset=offset,
        layer=layer,
        width_wide=width_wide,
        auto_widen=auto_widen,
        auto_widen_minimum_length=auto_widen_minimum_length,
        taper_length=taper_length,
        radius=radius,
        bbox_layers=bbox_layers or [],
        bbox_offsets=bbox_offsets or [],
        sections=sections or [],
        min_length=min_length,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        snap_to_grid=snap_to_grid,
        port_types=port_types,
        port_names=port_names,
        info=info or {},
        decorator=decorator,
    )


@pydantic.validate_arguments
def pin(
    width: float = 0.5,
    layer: Tuple[int, int] = LAYER.WG,
    layer_slab: Tuple[int, int] = LAYER.SLAB90,
    layers_via_stack1: Layers = (LAYER.PPP,),
    layers_via_stack2: Layers = (LAYER.NPP,),
    bbox_offsets_via_stack1: Tuple[float, ...] = (0, -0.2),
    bbox_offsets_via_stack2: Tuple[float, ...] = (0, -0.2),
    via_stack_width: float = 9.0,
    via_stack_gap: float = 0.55,
    slab_gap: float = -0.2,
    layer_via: Optional[Layer] = None,
    via_width: float = 1,
    via_offsets: Optional[Tuple[float, ...]] = None,
    **kwargs,
) -> CrossSection:
    """Rib PIN doped cross_section.

    Args:
        width: ridge width
        layer: ridge layer
        layer_slab: slab layer
        layers_via_stack1: P++ layer
        layers_via_stack2: N++ layer
        bbox_offsets_via_stack1:
        bbox_offsets_via_stack2:
        via_stack_width:
        via_stack_gap: offset from via_stack to ridge edge
        slab_gap: extra slab gap (negative: via_stack goes beyond slab)
        layer_via:
        via_width:
        via_offsets:
        kwargs: other cross_section settings

    https://doi.org/10.1364/OE.26.029983

    .. code::

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
    """
    slab_width = width + 2 * via_stack_gap + 2 * via_stack_width - 2 * slab_gap
    via_stack_offset = width / 2 + via_stack_gap + via_stack_width / 2

    sections = [Section(width=slab_width, layer=layer_slab, name="slab")]
    sections += [
        Section(
            layer=layer,
            width=via_stack_width + 2 * cladding_offset,
            offset=+via_stack_offset,
        )
        for layer, cladding_offset in zip(layers_via_stack1, bbox_offsets_via_stack1)
    ]
    sections += [
        Section(
            layer=layer,
            width=via_stack_width + 2 * cladding_offset,
            offset=-via_stack_offset,
        )
        for layer, cladding_offset in zip(layers_via_stack2, bbox_offsets_via_stack2)
    ]

    if layer_via and via_width and via_offsets:
        sections += [
            Section(
                layer=layer_via,
                width=via_width,
                offset=offset,
            )
            for offset in via_offsets
        ]
    info = dict(
        width=width,
        layer=layer,
        layer_slab=layer_slab,
        layers_via_stack1=layers_via_stack1,
        layers_via_stack2=layers_via_stack2,
        bbox_offsets_via_stack1=bbox_offsets_via_stack1,
        bbox_offsets_via_stack2=bbox_offsets_via_stack2,
        via_stack_width=via_stack_width,
        via_stack_gap=via_stack_gap,
        slab_gap=slab_gap,
        layer_via=layer_via,
        via_width=via_width,
        via_offsets=via_offsets,
        **kwargs,
    )

    return cross_section(
        width=width,
        layer=layer,
        sections=sections,
        info=info,
        **kwargs,
    )


@pydantic.validate_arguments
def pn(
    width: float = 0.5,
    layer: Tuple[int, int] = LAYER.WG,
    layer_slab: Tuple[int, int] = LAYER.SLAB90,
    gap_low_doping: float = 0.0,
    gap_medium_doping: Optional[float] = 0.5,
    gap_high_doping: Optional[float] = 1.0,
    width_doping: float = 8.0,
    width_slab: float = 7.0,
    layer_p: Tuple[int, int] = LAYER.P,
    layer_pp: Tuple[int, int] = LAYER.PP,
    layer_ppp: Tuple[int, int] = LAYER.PPP,
    layer_n: Tuple[int, int] = LAYER.N,
    layer_np: Tuple[int, int] = LAYER.NP,
    layer_npp: Tuple[int, int] = LAYER.NPP,
    port_names: Tuple[str, str] = ("o1", "o2"),
    bbox_layers: Optional[List[Layer]] = None,
    bbox_offsets: Optional[List[float]] = None,
) -> CrossSection:
    """Rib PN doped cross_section.

    Args:
        width: width of the ridge
        layer: ridge llayer
        layer_slab: slab layer
        gap_low_doping: from waveguide center to low doping
        gap_medium_doping: from waveguide center to medium doping.
            None removes medium doping
        gap_high_doping: from center to high doping. None removes it.
        width_doping:
        width_slab:
        layer_p:
        layer_pp:
        layer_ppp:
        layer_n:
        layer_np:
        layer_npp:
        bbox_layers: list of layers for rectangular bounding box.
        bbox_offsets: list of bounding box offsets.
        port_names:


    .. code::

                                   layer
                           |<------width------>|
                            ____________________
                           |     |       |     |
        ___________________|     |       |     |__________________________|
                    P            |       |              N                 |
                 width_p         |       |           width_n              |
        <----------------------->|       |<------------------------------>|
                                     |<->|
                                     gap_low_doping
                                     |         |        N+                |
                                     |         |     width_np             |
                                     |         |<------------------------>|
                                     |<------->|
                                           gap_medium_doping

    """
    slab = Section(width=width_slab, offset=0, layer=layer_slab)
    sections = [slab]
    offset_low_doping = width_doping / 2 + gap_low_doping
    width_low_doping = width_doping - gap_low_doping

    n = Section(width=width_low_doping, offset=+offset_low_doping, layer=layer_n)
    p = Section(width=width_low_doping, offset=-offset_low_doping, layer=layer_p)
    sections.append(n)
    sections.append(p)

    if gap_medium_doping is not None:
        width_medium_doping = width_doping - gap_medium_doping
        offset_medium_doping = width_medium_doping / 2 + gap_medium_doping

        np = Section(
            width=width_medium_doping,
            offset=+offset_medium_doping,
            layer=layer_np,
        )
        pp = Section(
            width=width_medium_doping,
            offset=-offset_medium_doping,
            layer=layer_pp,
        )
        sections.append(np)
        sections.append(pp)

    if gap_high_doping is not None:
        width_high_doping = width_doping - gap_high_doping
        offset_high_doping = width_high_doping / 2 + gap_high_doping
        npp = Section(
            width=width_high_doping, offset=+offset_high_doping, layer=layer_npp
        )
        ppp = Section(
            width=width_high_doping, offset=-offset_high_doping, layer=layer_ppp
        )
        sections.append(npp)
        sections.append(ppp)

    bbox_layers = bbox_layers or []
    bbox_offsets = bbox_offsets or []
    for layer_cladding, cladding_offset in zip(bbox_layers, bbox_offsets):
        s = Section(
            width=width_slab + 2 * cladding_offset, offset=0, layer=layer_cladding
        )
        sections.append(s)

    info = dict(
        width=width,
        layer=layer,
        bbox_layers=bbox_layers,
        bbox_offsets=bbox_offsets,
        gap_low_doping=gap_low_doping,
        gap_medium_doping=gap_medium_doping,
        gap_high_doping=gap_high_doping,
        width_doping=width_doping,
        width_slab=width_slab,
    )
    return CrossSection(
        width=width,
        offset=0,
        layer=layer,
        port_names=port_names,
        info=info,
        sections=sections,
    )


@pydantic.validate_arguments
def strip_heater_metal_undercut(
    width: float = 0.5,
    layer: Layer = LAYER.WG,
    heater_width: float = 2.5,
    trench_width: float = 6.5,
    trench_gap: float = 2.0,
    layer_heater: Layer = LAYER.HEATER,
    layer_trench: Layer = LAYER.DEEPTRENCH,
    **kwargs,
) -> CrossSection:
    """Returns strip cross_section with top metal and undercut trenches on both sides.
    dimensions from https://doi.org/10.1364/OE.18.020298

    Args:
        width: of waveguide
        layer:
        heater_width: of metal heater
        trench_width:
        trench_gap: from waveguide edge to trench edge
        layer_heater:
        layer_trench:
        kwargs: cross_section settings


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

    """
    trench_offset = trench_gap + trench_width / 2 + width / 2
    info = dict(
        width=width,
        layer=layer,
        heater_width=heater_width,
        trench_width=trench_width,
        trench_gap=trench_gap,
        layer_heater=layer_heater,
        layer_trench=layer_trench,
        **kwargs,
    )
    return cross_section(
        width=width,
        layer=layer,
        sections=(
            Section(layer=layer_heater, width=heater_width),
            Section(layer=layer_trench, width=trench_width, offset=+trench_offset),
            Section(layer=layer_trench, width=trench_width, offset=-trench_offset),
        ),
        info=info,
        **kwargs,
    )


@pydantic.validate_arguments
def strip_heater_metal(
    width: float = 0.5,
    layer: Layer = LAYER.WG,
    heater_width: float = 2.5,
    layer_heater: Layer = LAYER.HEATER,
    **kwargs,
) -> CrossSection:
    """Returns strip cross_section with top heater metal.
    dimensions from https://doi.org/10.1364/OE.18.020298

    Args:
        width: of waveguide
        layer:
        heater_width: of metal heater
        layer_heater: for the metal

    """
    info = dict(
        width=width,
        layer=layer,
        heater_width=heater_width,
        layer_heater=layer_heater,
        **kwargs,
    )

    return cross_section(
        width=width,
        layer=layer,
        sections=[Section(layer=layer_heater, width=heater_width)],
        info=info,
        **kwargs,
    )


@pydantic.validate_arguments
def heater_metal(
    width: float = 2.5,
    layer: Layer = LAYER.HEATER,
    **kwargs,
) -> CrossSection:
    """Returns metal heater cross_section.
    dimensions from https://doi.org/10.1364/OE.18.020298

    Args:
        width: of the metal
        layer: of the heater

    """
    return cross_section(
        width=width,
        layer=layer,
        **kwargs,
    )


@pydantic.validate_arguments
def strip_heater_doped(
    width: float = 0.5,
    layer: Layer = LAYER.WG,
    heater_width: float = 2.0,
    heater_gap: float = 0.8,
    layers_heater: Layers = (LAYER.WG, LAYER.NPP),
    bbox_offsets_heater: Tuple[float, ...] = (0, 0.1),
    **kwargs,
) -> CrossSection:
    """Returns strip cross_section with N++ doped heaters on both sides.

    .. code::

                                  |<------width------>|
          ____________             ___________________               ______________
         |            |           |     undoped Si    |             |              |
         |layer_heater|           |  intrinsic region |<----------->| layer_heater |
         |____________|           |___________________|             |______________|
                                                                     <------------>
                                                        heater_gap     heater_width
    """
    heater_offset = width / 2 + heater_gap + heater_width / 2

    sections = [
        Section(
            layer=layer,
            width=heater_width + 2 * cladding_offset,
            offset=+heater_offset,
        )
        for layer, cladding_offset in zip(layers_heater, bbox_offsets_heater)
    ]

    sections += [
        Section(
            layer=layer,
            width=heater_width + 2 * cladding_offset,
            offset=-heater_offset,
        )
        for layer, cladding_offset in zip(layers_heater, bbox_offsets_heater)
    ]

    return cross_section(
        width=width,
        layer=layer,
        sections=sections,
        **kwargs,
    )


strip_heater_doped_via_stack = partial(
    strip_heater_doped,
    layers_heater=(LAYER.WG, LAYER.NPP, LAYER.VIAC),
    bbox_offsets_heater=(0, 0.1, -0.2),
)


@pydantic.validate_arguments
def rib_heater_doped(
    width: float = 0.5,
    layer: Layer = LAYER.WG,
    heater_width: float = 2.0,
    heater_gap: float = 0.8,
    layer_heater: Layer = LAYER.NPP,
    layer_slab: Layer = LAYER.SLAB90,
    slab_gap: float = 0.2,
    with_top_heater: bool = True,
    with_bot_heater: bool = True,
    **kwargs,
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
    """
    heater_offset = width / 2 + heater_gap + heater_width / 2

    if with_bot_heater and with_top_heater:
        slab_width = width + 2 * heater_gap + 2 * heater_width + 2 * slab_gap
        slab_offset = 0
    elif with_top_heater:
        slab_width = width + heater_gap + heater_width + slab_gap
        slab_offset = -slab_width / 2
    elif with_bot_heater:
        slab_width = width + heater_gap + heater_width + slab_gap
        slab_offset = +slab_width / 2

    sections = []

    if with_bot_heater:
        sections += [
            Section(layer=layer_heater, width=heater_width, offset=+heater_offset)
        ]
    if with_top_heater:
        sections += [
            Section(layer=layer_heater, width=heater_width, offset=-heater_offset)
        ]
    sections += [
        Section(width=slab_width, layer=layer_slab, offset=slab_offset, name="slab")
    ]
    return cross_section(
        width=width,
        layer=layer,
        sections=sections,
        **kwargs,
    )


@pydantic.validate_arguments
def rib_heater_doped_via_stack(
    width: float = 0.5,
    layer: Layer = LAYER.WG,
    heater_width: float = 1.0,
    heater_gap: float = 0.8,
    layer_slab: Layer = LAYER.SLAB90,
    layer_heater: Layer = LAYER.NPP,
    via_stack_width: float = 2.0,
    via_stack_gap: float = 0.8,
    layers_via_stack: Layers = (LAYER.NPP, LAYER.VIAC),
    bbox_offsets_via_stack: Tuple[float, ...] = (0, -0.2),
    slab_gap: float = 0.2,
    slab_offset: float = 0,
    with_top_heater: bool = True,
    with_bot_heater: bool = True,
    **kwargs,
) -> CrossSection:
    """Returns rib cross_section with N++ doped heaters on both sides.
    dimensions from https://doi.org/10.1364/OE.27.010456

    Args:
        width:
        layer:
        heater_width:
        heater_gap:
        layer_slab:
        layer_heater:
        via_stack_width:
        via_stack_gap:
        layers_via_stack:
        bbox_offsets_via_stack:
        slab_gap: from heater edge
        slab_offset: over the center of the slab
        with_top_heater:
        with_bot_heater:

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

    """
    if with_bot_heater and with_top_heater:
        slab_width = width + 2 * heater_gap + 2 * heater_width + 2 * slab_gap
    elif with_top_heater:
        slab_width = width + heater_gap + heater_width + slab_gap
        slab_offset -= slab_width / 2
    elif with_bot_heater:
        slab_width = width + heater_gap + heater_width + slab_gap
        slab_offset += slab_width / 2

    heater_offset = width / 2 + heater_gap + heater_width / 2
    via_stack_offset = width / 2 + via_stack_gap + via_stack_width / 2
    sections = [
        Section(width=slab_width, layer=layer_slab, offset=slab_offset, name="slab"),
    ]
    if with_bot_heater:
        sections += [
            Section(
                layer=layer_heater,
                width=heater_width,
                offset=+heater_offset,
            )
        ]

    if with_top_heater:
        sections += [
            Section(
                layer=layer_heater,
                width=heater_width,
                offset=-heater_offset,
            )
        ]

    if with_bot_heater:
        sections += [
            Section(
                layer=layer,
                width=heater_width + 2 * cladding_offset,
                offset=+via_stack_offset,
            )
            for layer, cladding_offset in zip(layers_via_stack, bbox_offsets_via_stack)
        ]

    if with_top_heater:
        sections += [
            Section(
                layer=layer,
                width=heater_width + 2 * cladding_offset,
                offset=-via_stack_offset,
            )
            for layer, cladding_offset in zip(layers_via_stack, bbox_offsets_via_stack)
        ]

    return cross_section(
        sections=sections,
        width=width,
        layer=layer,
        **kwargs,
    )


# xs_optical = partial(
#     cross_section, layer_bbox=(68, 0), decorator=add_pins_siepic_optical
# )
# strip = partial(cross_section, layer_bbox=(68, 0), decorator=add_pins_siepic_optical)

strip = partial(cross_section)
strip_auto_widen = partial(cross_section, width_wide=0.9, auto_widen=True)
rib = partial(
    strip,
    sections=[Section(width=6, layer=LAYER.SLAB90, name="slab")],
    bbox_layers=[LAYER.SLAB90],
    bbox_offsets=[3],
)
nitride = partial(cross_section, layer=LAYER.WGN, width=1.0)
strip_rib_tip = partial(
    strip, sections=(Section(width=0.2, layer=LAYER.SLAB90, name="slab"),)
)

port_names_electrical = ("e1", "e2")
port_types_electrical = ("electrical", "electrical")
metal1 = partial(
    cross_section,
    layer=LAYER.M1,
    width=10.0,
    port_names=port_names_electrical,
    port_types=port_types_electrical,
)
metal2 = partial(
    metal1,
    layer=LAYER.M2,
)
metal3 = partial(
    metal1,
    layer=LAYER.M3,
)


CrossSectionFactory = Callable[..., CrossSection]


def get_cross_section_factories(
    modules, verbose: bool = False
) -> Dict[str, CrossSectionFactory]:
    """Returns cross_section factories from a module or list of modules.

    Args:
        modules: module or iterable of modules.
        verbose: prints in case any errors occur.
    """

    modules = modules if isinstance(modules, Iterable) else [modules]

    xs = {}
    for module in modules:
        for t in getmembers(module):
            if callable(t[1]) and t[0] != "partial":
                try:
                    r = inspect.signature(t[1]).return_annotation
                    if r == CrossSection:
                        xs[t[0]] = t[1]
                except ValueError:
                    if verbose:
                        print(f"error in {t[0]}")
    return xs


cross_sections = get_cross_section_factories(sys.modules[__name__])

if __name__ == "__main__":
    import gdsfactory as gf

    p = gf.path.straight()
    x = CrossSection(name="strip", layer=(1, 0), width=0.5)
    x = x.get_copy(width=3)
    c = p.extrude(x)
    c.show()

    # P = gf.path.euler(radius=10, use_eff=True)
    # P = euler()
    # P = gf.Path()
    # P.append(gf.path.straight(length=5))
    # P.append(gf.path.arc(radius=10, angle=90))
    # P.append(gf.path.spiral())

    # Create a blank CrossSection

    # X = pin(width=0.5, width_i=0.5)
    # x = strip(width=0.5)

    # X = strip_heater_metal_undercut()
    # X = metal1()
    # X = pin(layer_via=LAYER.VIAC, via_offsets=(-2, 2))
    # X = pin()
    # X = strip_heater_doped()

    # x1 = strip_rib_tip()
    # x2 = rib_heater_doped_via_stack()
    # X = gf.path.transition(x1, x2)
    # P = gf.path.straight(npoints=100, length=10)

    # X = CrossSection()

    # X = rib_heater_doped(with_bot_heater=False, decorator=add_pins_siepic_optical)
    # P = gf.path.straight(npoints=100, length=10)
    # c = gf.path.extrude(P, X)

    # print(x1.to_dict())
    # print(x1.name)
    # c = gf.path.component(P, strip(width=2, layer=LAYER.WG, cladding_offset=3))
    # c = gf.add_pins(c)
    # c << gf.components.bend_euler(radius=10)
    # c << gf.components.bend_circular(radius=10)
    # c.pprint_ports()
    # c.show(show_ports=False)
