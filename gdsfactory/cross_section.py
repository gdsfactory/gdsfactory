"""You can define a path with a list of points

To create a component you need to extrude the path with a cross-section.

Based on phidl.device_layout.CrossSection
"""
from functools import partial
from typing import Optional, Tuple

import pydantic
from phidl.device_layout import CrossSection as CrossSectionPhidl

from gdsfactory.tech import TECH, Section

LAYER = TECH.layer
Layer = Tuple[int, int]
Layers = Tuple[Layer, ...]
Floats = Tuple[float, ...]


class CrossSection(CrossSectionPhidl):
    """Add port_types to phidl cross_section

    .. code::


          0   offset
          |<-------------->|
          |              _____
          |             |     |
          |             |layer|
          |             |_____|
          |              <---->
                         width


    """

    def __init__(self):
        self.sections = []
        self.ports = set()
        self.port_types = set()
        self.aliases = {}
        self.info = {}
        self.name = None

    def add(
        self,
        width: float = 1,
        offset: float = 0,
        layer: Tuple[int, int] = (1, 0),
        ports: Tuple[Optional[str], Optional[str]] = (None, None),
        name: Optional[str] = None,
        port_types: Tuple[str, str] = ("optical", "optical"),
        hidden: bool = False,
    ):
        """Adds a cross-sectional element to the CrossSection.  If ports are
        specified, when creating a Device with the extrude() command there be
        have Ports at the ends.

        Args:
            width: Width of the segment
            offset: Offset of the segment (positive values = right hand side)
            layer: The polygon layer to put the segment on
            ports: port names at the ends of the cross-section
            name: Name of the cross-sectional element for later access
            port_types: electrical, optical ...
            hidden: if True does not draw polygons for CrossSection
        """
        if isinstance(width, (float, int)) and (width <= 0):
            raise ValueError("CrossSection.add(): widths must be >0")
        if len(ports) != 2:
            raise ValueError("CrossSection.add(): must receive 2 port names")
        for p in ports:
            if p is not None and p in self.ports:
                raise ValueError(
                    f"CrossSection.add(): a port named {p!r} already "
                    "exists in this CrossSection, please rename port"
                )

        [self.ports.add(port) for port in ports if port is not None]
        [self.port_types.add(port_type) for port_type in port_types]

        if name in self.aliases:
            raise ValueError(
                f"CrossSection.add(): an element named {name!r} already "
                "exists in this CrossSection, please change the name"
            )

        new_segment = dict(
            width=width,
            offset=offset,
            layer=layer,
            ports=ports,
            port_types=port_types,
            hidden=hidden,
            name=name,
        )

        if name is not None:
            self.aliases[name] = new_segment
        self.sections.append(new_segment)
        return self

    def copy(self):
        """Returns a copy of the CrossSection"""
        X = CrossSection()
        X.info = self.info.copy()
        X.sections = list(self.sections)
        X.ports = tuple(self.ports)
        X.aliases = dict(self.aliases)
        X.port_types = tuple(self.port_types)
        return X

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """pydantic assumes CrossSection is always valid"""
        assert isinstance(
            v, CrossSection
        ), f"TypeError, Got {type(v)}, expecting CrossSection"
        return v

    def to_dict(self):
        d = {}
        x = self.copy()
        d["sections"] = [dict(section) for section in x.sections if section]
        d["ports"] = x.ports
        d["port_types"] = x.port_types
        d["aliases"] = x.aliases
        d["info"] = x.info
        return d

    # @property
    # def name(self):
    #     return "_".join([str(i) for i in self.to_dict()["sections"]])

    def get_name(self):
        return self.name or "_".join([str(i) for i in self.to_dict()["sections"]])


@pydantic.validate_arguments
def cross_section(
    width: float = 0.5,
    layer: Tuple[int, int] = (1, 0),
    width_wide: Optional[float] = None,
    auto_widen: bool = False,
    auto_widen_minimum_length: float = 200.0,
    taper_length: float = 10.0,
    radius: float = 10.0,
    cladding_offset: float = 3.0,
    layers_cladding: Optional[Tuple[Layer, ...]] = None,
    sections: Optional[Tuple[Section, ...]] = None,
    port_names: Tuple[str, str] = ("o1", "o2"),
    port_types: Tuple[str, str] = ("optical", "optical"),
    min_length: float = 10e-3,
    start_straight_length: float = 10e-3,
    end_straight_length: float = 10e-3,
    snap_to_grid: Optional[float] = None,
) -> CrossSection:
    """Returns CrossSection.

    Args:
        width: main of the waveguide
        layer: main layer
        width_wide: taper to widen waveguides for lower loss
        auto_widen: taper to widen waveguides for lower loss
        auto_widen_minimum_length: minimum straight length for auto_widen
        taper_length: taper_length for auto_widen
        radius: bend radius
        cladding_offset: offset for layers_cladding
        layers_cladding:
        sections: Sections(width, offset, layer, ports)
        port_names: for input and output (1, 2),
        port_types: for input and output (electrical, optical, vertical_te ...)
        min_length: 10e-3 for routing
        start_straight_length: for routing
        end_straight_length: for routing
        snap_to_grid: can snap points to grid when extruding the path
    """

    x = CrossSection()
    x.add(
        width=width,
        offset=0,
        layer=layer,
        ports=port_names,
        port_types=port_types,
        name="_default",
    )

    sections = sections or []
    for section in sections:
        if isinstance(section, dict):
            x.add(
                width=section["width"],
                offset=section["offset"],
                layer=section["layer"],
                ports=section["ports"],
                port_types=section["port_types"],
                name=section["name"],
            )
        else:
            x.add(
                width=section.width,
                offset=section.offset,
                layer=section.layer,
                ports=section.ports,
                port_types=section.port_types,
                name=section.name,
            )

    x.info = dict(
        width=width,
        layer=layer,
        width_wide=width_wide,
        auto_widen=auto_widen,
        auto_widen_minimum_length=auto_widen_minimum_length,
        taper_length=taper_length,
        radius=radius,
        cladding_offset=cladding_offset,
        layers_cladding=layers_cladding,
        sections=sections,
        min_length=min_length,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        snap_to_grid=snap_to_grid,
        port_types=port_types,
        port_names=port_names,
    )
    return x


@pydantic.validate_arguments
def pin(
    width: float = 0.5,
    layer: Tuple[int, int] = LAYER.WG,
    layer_slab: Tuple[int, int] = LAYER.SLAB90,
    layers_contact1: Layers = (LAYER.PPP,),
    layers_contact2: Layers = (LAYER.NPP,),
    cladding_offsets_contact1: Tuple[float, ...] = (0, -0.2),
    cladding_offsets_contact2: Tuple[float, ...] = (0, -0.2),
    contact_width: float = 9.0,
    contact_gap: float = 0.55,
    slab_gap: float = -0.2,
    layer_via: Optional[Layer] = None,
    via_width: float = 1,
    via_offsets: Optional[Tuple[float, ...]] = None,
    **kwargs,
) -> CrossSection:
    """rib PIN doped cross_section.

    Args:
        width: ridge width
        layer: ridge layer
        layer_slab: slab layer
        layers_contact1: P++ layer
        layers_contact2: N++ layer
        cladding_offsets_contact1:
        cladding_offsets_contact2:
        contact_width:
        contact_gap: offset from contact to ridge edge
        slab_gap: extra slab gap (negative: contact goes beyond slab)
        layer_via:
        via_width:
        via_offsets:
        kwargs: other cross_section settings

    https://doi.org/10.1364/OE.26.029983

    .. code::

                                         layer
                                 |<------width------>|
                                  ____________________ contact_gap              slab_gap
                                 |                   |<----------->|             <-->
        ___ _____________________|                   |__________________________|___
       |   |         |                 undoped Si                  |            |   |
       |   |    P++  |                 intrinsic region            |     N++    |   |
       |___|_________|_____________________________________________|____________|___|
                                                                    <----------->
                                                                    contact_width
       <---------------------------------------------------------------------------->
                                      slab_width
    """
    slab_width = width + 2 * contact_gap + 2 * contact_width - 2 * slab_gap
    contact_offset = width / 2 + contact_gap + contact_width / 2

    sections = [Section(width=slab_width, layer=layer_slab, name="slab")]
    sections += [
        Section(
            layer=layer,
            width=contact_width + 2 * cladding_offset,
            offset=+contact_offset,
        )
        for layer, cladding_offset in zip(layers_contact1, cladding_offsets_contact1)
    ]
    sections += [
        Section(
            layer=layer,
            width=contact_width + 2 * cladding_offset,
            offset=-contact_offset,
        )
        for layer, cladding_offset in zip(layers_contact2, cladding_offsets_contact2)
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

    x = cross_section(
        width=width,
        layer=layer,
        sections=sections,
        **kwargs,
    )
    info = dict(
        width=width,
        layer=layer,
        layer_slab=layer_slab,
        layers_contact1=layers_contact1,
        layers_contact2=layers_contact2,
        cladding_offsets_contact1=cladding_offsets_contact1,
        cladding_offsets_contact2=cladding_offsets_contact2,
        contact_width=contact_width,
        contact_gap=contact_gap,
        slab_gap=slab_gap,
        layer_via=layer_via,
        via_width=via_width,
        via_offsets=via_offsets,
    )
    x.info.update(**info)
    x.info.update(**kwargs)
    return x


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
    cladding_offsets: Floats = (0,),
    layers_cladding: Optional[Tuple[Layer, ...]] = None,
    port_names: Tuple[str, str] = ("o1", "o2"),
) -> CrossSection:
    """rib PN doped cross_section.

    Args:
        width: width of the ridge
        layer: ridge llayer
        layer_slab: slab layer
        gap_low_doping: from waveguide center to low doping
        gap_medium_doping: from waveguide center to medium doping. None removes medium doping
        gap_high_doping: from waveguide center to high doping. None removes high doping
        width_doping:
        width_slab:
        layer_p:
        layer_pp:
        layer_ppp:
        layer_n:
        layer_np:
        layer_npp:
        cladding_offsets:
        layers_cladding: Iterable of layers
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
    x = CrossSection()
    x.add(width=width, offset=0, layer=layer, ports=port_names)
    x.add(width=width_slab, offset=0, layer=layer_slab)

    offset_low_doping = width_doping / 2 + gap_low_doping
    width_low_doping = width_doping - gap_low_doping
    x.add(width=width_low_doping, offset=+offset_low_doping, layer=layer_n)
    x.add(width=width_low_doping, offset=-offset_low_doping, layer=layer_p)

    if gap_medium_doping is not None:
        width_medium_doping = width_doping - gap_medium_doping
        offset_medium_doping = width_medium_doping / 2 + gap_medium_doping

        x.add(
            width=width_medium_doping,
            offset=+offset_medium_doping,
            layer=layer_np,
        )
        x.add(
            width=width_medium_doping,
            offset=-offset_medium_doping,
            layer=layer_pp,
        )

    if gap_high_doping is not None:
        width_high_doping = width_doping - gap_high_doping
        offset_high_doping = width_high_doping / 2 + gap_high_doping
        x.add(width=width_high_doping, offset=+offset_high_doping, layer=layer_npp)
        x.add(width=width_high_doping, offset=-offset_high_doping, layer=layer_ppp)

    layers_cladding = layers_cladding or []

    for cladding_offset, layer_cladding in zip(cladding_offsets, layers_cladding):
        x.add(width=width_slab + 2 * cladding_offset, offset=0, layer=layer_cladding)

    s = dict(
        width=width,
        layer=layer,
        cladding_offsets=cladding_offsets,
        layers_cladding=layers_cladding,
        gap_low_doping=gap_low_doping,
        gap_medium_doping=gap_medium_doping,
        gap_high_doping=gap_high_doping,
        width_doping=width_doping,
        width_slab=width_slab,
    )
    x.info = s
    return x


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
        **kwargs: for cross_section


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
    x = cross_section(
        width=width,
        layer=layer,
        sections=(
            Section(layer=layer_heater, width=heater_width),
            Section(layer=layer_trench, width=trench_width, offset=+trench_offset),
            Section(layer=layer_trench, width=trench_width, offset=-trench_offset),
        ),
        **kwargs,
    )
    x.info["heater_width"] = heater_width
    x.info["layer_heater"] = layer_heater
    x.info["trench_width"] = trench_width
    x.info["layer_heater"] = layer_heater
    x.info["trench_gap"] = trench_gap
    return x


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
    x = cross_section(
        width=width,
        layer=layer,
        sections=(Section(layer=layer_heater, width=heater_width),),
        **kwargs,
    )
    x.info["heater_width"] = heater_width
    x.info["layer_heater"] = layer_heater
    return x


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
    cladding_offsets_heater: Tuple[float, ...] = (0, 0.1),
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
        for layer, cladding_offset in zip(layers_heater, cladding_offsets_heater)
    ]

    sections += [
        Section(
            layer=layer,
            width=heater_width + 2 * cladding_offset,
            offset=-heater_offset,
        )
        for layer, cladding_offset in zip(layers_heater, cladding_offsets_heater)
    ]

    return cross_section(
        width=width,
        layer=layer,
        sections=sections,
        **kwargs,
    )


strip_heater_doped_contact = partial(
    strip_heater_doped,
    layers_heater=(LAYER.WG, LAYER.NPP, LAYER.VIAC),
    cladding_offsets_heater=(0, 0.1, -0.2),
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
                                      ____________________  heater_gap                slab_gap
                                     |                   |<----------->|               <-->
         ___ ________________________|                   |____________________________|___
        |   |            |                 undoped Si                  |              |   |
        |   |layer_heater|                 intrinsic region            |layer_heater  |   |
        |___|____________|_____________________________________________|______________|___|
                                                                        <------------>
                                                                         heater_width
        <--------------------------------------------------------------------------------->
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
def rib_heater_doped_contact(
    width: float = 0.5,
    layer: Layer = LAYER.WG,
    heater_width: float = 1.0,
    heater_gap: float = 0.8,
    layer_slab: Layer = LAYER.SLAB90,
    layer_heater: Layer = LAYER.NPP,
    contact_width: float = 2.0,
    contact_gap: float = 0.8,
    layers_contact: Layers = (LAYER.NPP, LAYER.VIAC),
    cladding_offsets_contact: Tuple[float, ...] = (0, -0.2),
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
        contact_width:
        contact_gap:
        layers_contact:
        cladding_offsets_contact:
        slab_gap: from heater edge
        slab_offset: over the center of the slab
        with_top_heater:
        with_bot_heater:

    .. code::

                                    |<------width------>|
       slab_gap                      ____________________  contact_gap     contact width
       <-->                         |                   |<------------->|<----------------->
                                    |                   |  heater_gap |
                                    |                   |<----------->|
        ___ ________________________|                   |____________________________ ______
       |   |            |                 undoped Si                  |              |      |
       |   |layer_heater|                 intrinsic region            |layer_heater  |      |
       |___|____________|_____________________________________________|______________|______|
                                                                       <------------>
                                                                        heater_width
       <--------------------------------------------------------------------------------->
                                         slab_width

    """
    if with_bot_heater and with_top_heater:
        slab_width = width + 2 * heater_gap + 2 * heater_width + 2 * slab_gap
        slab_offset = slab_offset
    elif with_top_heater:
        slab_width = width + heater_gap + heater_width + slab_gap
        slab_offset = slab_offset - slab_width / 2
    elif with_bot_heater:
        slab_width = width + heater_gap + heater_width + slab_gap
        slab_offset = slab_offset + slab_width / 2

    heater_offset = width / 2 + heater_gap + heater_width / 2
    contact_offset = width / 2 + contact_gap + contact_width / 2
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
                offset=+contact_offset,
            )
            for layer, cladding_offset in zip(layers_contact, cladding_offsets_contact)
        ]

    if with_top_heater:
        sections += [
            Section(
                layer=layer,
                width=heater_width + 2 * cladding_offset,
                offset=-contact_offset,
            )
            for layer, cladding_offset in zip(layers_contact, cladding_offsets_contact)
        ]

    return cross_section(
        sections=sections,
        width=width,
        layer=layer,
        **kwargs,
    )


strip = partial(cross_section)
strip_auto_widen = partial(cross_section, width_wide=0.9, auto_widen=True)
rib = partial(
    strip,
    sections=(Section(width=6, layer=LAYER.SLAB90, name="slab"),),
    layers_cladding=(LAYER.SLAB90,),
    cladding_offset=0,
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
    width=10.0,
    port_names=port_names_electrical,
    port_types=port_types_electrical,
)
metal3 = partial(
    metal1,
    layer=LAYER.M3,
    width=10.0,
    port_names=port_names_electrical,
    port_types=port_types_electrical,
)


# xs_strip = strip()
# xs_strip_auto_widen = strip_auto_widen()
# xs_rib = rib()
# xs_nitride = nitride()
# xs_metal1 = metal1()
# xs_metal2 = metal2()
# xs_metal3 = metal3()
# xs_pin = pin()
# xs_strip_heater_metal_undercut = strip_heater_metal_undercut()
# xs_strip_heater_metal = strip_heater_metal()
# xs_strip_heater_doped = strip_heater_doped()
# xs_rib_heater_doped = rib_heater_doped()


cross_section_factory = dict(
    cross_section=cross_section,
    strip=strip,
    strip_auto_widen=strip_auto_widen,
    rib=rib,
    nitride=nitride,
    metal1=metal1,
    metal2=metal2,
    metal3=metal3,
    pin=pin,
    strip_heater_metal_undercut=strip_heater_metal_undercut,
    strip_heater_metal=strip_heater_metal,
    strip_heater_doped=strip_heater_doped,
    rib_heater_doped=rib_heater_doped,
)

if __name__ == "__main__":
    import gdsfactory as gf

    P = gf.path.straight()
    # P = gf.path.euler(radius=10, use_eff=True)
    # P = euler()
    # P = gf.Path()
    # P.append(gf.path.straight(length=5))
    # P.append(gf.path.arc(radius=10, angle=90))
    # P.append(gf.path.spiral())

    # Create a blank CrossSection
    # X.add(width=2.0, offset=4, layer=LAYER.HEATER, ports=["HW0", "HE0"])

    # X = pin(width=0.5, width_i=0.5)
    # x = strip(width=0.5)

    # X = strip_heater_metal_undercut()
    # X = metal1()
    # X = pin(layer_via=LAYER.VIAC, via_offsets=(-2, 2))
    # X = pin()
    # X = strip_heater_doped()

    # x1 = strip_rib_tip()
    # x2 = rib_heater_doped_contact()
    # X = gf.path.transition(x1, x2)
    # P = gf.path.straight(npoints=100, length=10)

    # X = CrossSection()
    # X.add(width=2.0, offset=-4, layer=LAYER.HEATER, ports=["e1", "e2"])
    # X.add(width=0.5, offset=0, layer=LAYER.SLAB90, ports=["o1", "o2"])

    X = rib_heater_doped(with_bot_heater=False)
    P = gf.path.straight(npoints=100, length=10)

    c = gf.path.extrude(P, X)

    # print(x1.to_dict())
    # print(x1.name)

    # c = gf.path.component(P, strip(width=2, layer=LAYER.WG, cladding_offset=3))
    # c = gf.add_pins(c)
    # c << gf.components.bend_euler(radius=10)
    # c << gf.components.bend_circular(radius=10)
    # c.pprint_ports()
    c.show()
