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


class CrossSection(CrossSectionPhidl):
    def __init__(self):
        self.sections = []
        self.ports = (None, None)
        self.port_types = (None, None)
        self.aliases = {}
        self.info = {}

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
            ports: If not None, specifies the names for the ports at the ends of the
                cross-sectional element
            name: Name of the cross-sectional element for later access
            port_types: port of the cross types
            hidden: if True does not draw polygons for CrossSection
        """
        if isinstance(width, (float, int)) and (width <= 0):
            raise ValueError("CrossSection.add(): widths must be >0")
        if len(ports) != 2:
            raise ValueError("CrossSection.add(): must receive 2 port names")
        for i, p in enumerate(ports):
            if p is not None and p in self.ports:
                raise ValueError(
                    f"CrossSection.add(): a port named {p} already "
                    "exists in this CrossSection, please rename port"
                )
            if p is not None:
                if self.ports[i] is None:
                    new_ports = list(self.ports)
                    new_ports[i] = p
                    self.ports = tuple(new_ports)

                    new_ports_types = list(self.port_types)
                    new_ports_types[i] = port_types[i]
                    self.port_types = tuple(new_ports_types)
                else:
                    raise ValueError(f"Multiple ports defined in index {i}")

        if name in self.aliases:
            raise ValueError(
                'CrossSection.add(): an element named "%s" already '
                "exists in this CrossSection, please change the name" % name
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
    start_straight: float = 10e-3,
    end_straight_offset: float = 10e-3,
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
        min_length: 10e-3 for routing
        start_straight: for routing
        end_straight_offset: for routing
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
        start_straight=start_straight,
        end_straight_offset=end_straight_offset,
        snap_to_grid=snap_to_grid,
        port_types=port_types,
    )
    return x


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

    return cross_section(
        width=width,
        layer=layer,
        sections=sections,
        **kwargs,
    )


def pn(
    width: float = 0.5,
    layer: Tuple[int, int] = LAYER.WG,
    layer_slab: Tuple[int, int] = LAYER.SLAB90,
    width_i: float = 0.0,
    width_p: float = 1.0,
    width_n: float = 1.0,
    width_pp: float = 1.0,
    width_np: float = 1.0,
    width_ppp: float = 1.0,
    width_npp: float = 1.0,
    layer_p: Tuple[int, int] = LAYER.P,
    layer_n: Tuple[int, int] = LAYER.N,
    layer_pp: Tuple[int, int] = LAYER.PP,
    layer_np: Tuple[int, int] = LAYER.NP,
    layer_ppp: Tuple[int, int] = LAYER.PPP,
    layer_npp: Tuple[int, int] = LAYER.NPP,
    cladding_offset: float = 0,
    layers_cladding: Optional[Tuple[Layer, ...]] = None,
    port_names: Tuple[str, str] = ("o1", "o2"),
    **kwargs,
) -> CrossSection:
    """rib PN doped cross_section.

    .. code::

                                   layer
                           |<------width------>|
                            ____________________
                           |     |       |     |
        ___________________|     |       |     |__________________________|
                                 |       |                                |
            P++     P+     P     |   I   |     N        N+         N++    |
        _________________________|_______|________________________________|
                                                                          |
                                 |width_i| width_n | width_np | width_npp |
                                    0    oi        on        onp         onpp

    """
    x = CrossSection()
    x.add(width=width, offset=0, layer=layer, ports=port_names)

    oi = width_i / 2
    on = oi + width_n
    onp = oi + width_n + width_np
    onpp = oi + width_n + width_np + width_npp

    op = -oi - width_p
    opp = op - width_pp
    oppp = opp - width_ppp

    offset_n = (oi + on) / 2
    offset_np = (on + onp) / 2
    offset_npp = (onp + onpp) / 2

    offset_p = (-oi + op) / 2
    offset_pp = (op + opp) / 2
    offset_ppp = (opp + oppp) / 2

    width_slab = abs(onpp) + abs(oppp)
    x.add(width=width_slab, offset=0, layer=layer_slab)

    x.add(width=width_n, offset=offset_n, layer=layer_n)
    x.add(width=width_np, offset=offset_np, layer=layer_np)
    x.add(width=width_npp, offset=offset_npp, layer=layer_npp)

    x.add(width=width_p, offset=offset_p, layer=layer_p)
    x.add(width=width_pp, offset=offset_pp, layer=layer_pp)
    x.add(width=width_ppp, offset=offset_ppp, layer=layer_ppp)

    for layer_cladding in layers_cladding or []:
        x.add(width=width_slab + 2 * cladding_offset, offset=0, layer=layer_cladding)

    s = dict(
        width=width,
        layer=layer,
        cladding_offset=cladding_offset,
        layers_cladding=layers_cladding,
    )
    x.info = s
    x.info.update(**kwargs)
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
):
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
    return cross_section(
        width=width,
        layer=layer,
        sections=(
            Section(layer=layer_heater, width=heater_width),
            Section(layer=layer_trench, width=trench_width, offset=+trench_offset),
            Section(layer=layer_trench, width=trench_width, offset=-trench_offset),
        ),
        **kwargs,
    )


@pydantic.validate_arguments
def strip_heater_metal(
    width: float = 0.5,
    layer: Layer = LAYER.WG,
    heater_width: float = 2.5,
    layer_heater: Layer = LAYER.HEATER,
    **kwargs,
):
    """Returns strip cross_section with top heater metal.
    dimensions from https://doi.org/10.1364/OE.18.020298

    Args:
        width: of waveguide
        layer:
        heater_width: of metal heater
        layer_heater: for the metal

    """
    return cross_section(
        width=width,
        layer=layer,
        sections=(Section(layer=layer_heater, width=heater_width),),
        **kwargs,
    )


@pydantic.validate_arguments
def heater_metal(
    width: float = 2.5,
    layer: Layer = LAYER.HEATER,
    **kwargs,
):
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
):
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
    **kwargs,
):
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
    slab_width = width + 2 * heater_gap + 2 * heater_width + 2 * slab_gap
    return cross_section(
        width=width,
        layer=layer,
        sections=(
            Section(
                layer=layer_heater,
                width=heater_width,
                offset=+heater_offset,
            ),
            Section(
                layer=layer_heater,
                width=heater_width,
                offset=-heater_offset,
            ),
            Section(width=slab_width, layer=layer_slab, name="slab"),
        ),
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
    **kwargs,
):
    """Returns rib cross_section with N++ doped heaters on both sides.
    dimensions from https://doi.org/10.1364/OE.27.010456

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
    slab_width = width + 2 * heater_gap + 2 * heater_width + 2 * slab_gap
    heater_offset = width / 2 + heater_gap + heater_width / 2
    contact_offset = width / 2 + contact_gap + contact_width / 2
    sections = [
        Section(
            layer=layer_heater,
            width=heater_width,
            offset=+heater_offset,
        ),
        Section(
            layer=layer_heater,
            width=heater_width,
            offset=-heater_offset,
        ),
        Section(width=slab_width, layer=layer_slab, name="slab"),
    ]

    sections += [
        Section(
            layer=layer,
            width=heater_width + 2 * cladding_offset,
            offset=+contact_offset,
        )
        for layer, cladding_offset in zip(layers_contact, cladding_offsets_contact)
    ]

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
rib = partial(strip, sections=(Section(width=6, layer=LAYER.SLAB90, name="slab"),))
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
    # X = CrossSection()
    # X.add(width=2.0, offset=-4, layer=LAYER.HEATER, ports=["HW1", "HE1"])
    # X.add(width=0.5, offset=0, layer=LAYER.SLAB90, ports=["in", "out"])
    # X.add(width=2.0, offset=4, layer=LAYER.HEATER, ports=["HW0", "HE0"])

    # X = pin(width=0.5, width_i=0.5)
    # x = strip(width=0.5)

    # X = cross_section(width=3, layer=(2, 0))
    # X = cross_section(**s)
    # X = strip_heater_metal_undercut()
    # X = rib_heater_doped()

    # X = strip_heater_metal_undercut()
    # X = metal1()
    # X = pin(layer_via=LAYER.VIAC, via_offsets=(-2, 2))
    # X = pin()
    # X = strip_heater_doped()

    x1 = strip_rib_tip()
    x2 = rib_heater_doped_contact()
    X = gf.path.transition(x1, x2)
    P = gf.path.straight(npoints=100, length=10)

    c = gf.path.extrude(P, X)

    # c = gf.path.component(P, strip(width=2, layer=LAYER.WG, cladding_offset=3))
    # c = gf.add_pins(c)
    # c << gf.components.bend_euler(radius=10)
    # c << gf.components.bend_circular(radius=10)
    # c.pprint_ports
    c.show()
