"""Primitives."""

import gdsfactory as gf
from gdsfactory.cross_section import port_names_electrical, port_types_electrical
from gdsfactory.typings import CrossSectionSpec, LayerSpec, Size


@gf.cell
def straight(
    length: float = 10,
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
    npoints: int = 2,
) -> gf.Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        cross_section: specification (CrossSection, string or dict).
        width: width of the waveguide. If None, it will use the width of the cross_section.
        npoints: number of points.
    """
    return gf.c.straight(
        length=length, cross_section=cross_section, width=width, npoints=npoints
    )


@gf.cell
def straight_strip(
    length: float = 10,
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
    npoints: int = 2,
) -> gf.Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        cross_section: specification (CrossSection, string or dict).
        width: width of the waveguide. If None, it will use the width of the cross_section.
        npoints: number of points.
    """
    return gf.c.straight(
        length=length, cross_section=cross_section, width=width, npoints=npoints
    )


@gf.cell
def straight_rib(
    length: float = 10,
    cross_section: CrossSectionSpec = "rib",
    width: float | None = None,
) -> gf.Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        cross_section: specification (CrossSection, string or dict).
        width: width of the waveguide. If None, it will use the width of the cross_section.
    """
    return gf.c.straight(
        length=length, cross_section=cross_section, width=width, npoints=2
    )


@gf.cell
def bend_euler(
    radius: float | None = None,
    angle: float = 90,
    p: float = 0.5,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
) -> gf.Component:
    """Regular degree euler bend.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: total angle of the curve.
        p: Proportion of the curve that is an Euler curve.
        width: width to use. Defaults to cross_section.width.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
    """
    return gf.c.bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        with_arc_floorplan=True,
        npoints=None,
        layer=None,
    )


@gf.cell
def bend_s(
    size: Size = (11, 1.8),
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
    allow_min_radius_violation: bool = False,
    npoints: int = 99,
) -> gf.Component:
    """Return S bend with bezier curve.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: in x and y direction.
        cross_section: spec.
        width: width of the waveguide. If None, it will use the width of the cross_section.
        allow_min_radius_violation: allows min radius violations.
        npoints: number of points.
    """
    return gf.c.bend_s(
        size=size,
        cross_section=cross_section,
        npoints=npoints,
        allow_min_radius_violation=allow_min_radius_violation,
        width=width,
    )


@gf.cell
def wire_corner(
    cross_section: CrossSectionSpec = "metal_routing", width: float | None = None
) -> gf.Component:
    """Returns 45 degrees electrical corner wire.

    Args:
        cross_section: spec.
        width: optional width. Defaults to cross_section width.
    """
    return gf.c.wire_corner(
        cross_section=cross_section,
        width=width,
        port_names=port_names_electrical,
        port_types=port_types_electrical,
        radius=None,
    )


@gf.cell
def wire_corner45(
    cross_section: CrossSectionSpec = "metal_routing",
    radius: float = 10,
    width: float | None = None,
    layer: LayerSpec | None = None,
    with_corner90_ports: bool = True,
) -> gf.Component:
    """Returns 90 degrees electrical corner wire.

    Args:
        cross_section: spec.
        radius: ignored.
        width: optional width. Defaults to cross_section width.
        layer: ignored.
        with_corner90_ports: if True, adds ports at 90 degrees.
    """
    return gf.c.wire_corner45(
        cross_section=cross_section,
        radius=radius,
        width=width,
        layer=layer,
        with_corner90_ports=with_corner90_ports,
    )


####################
# Metal waveguides
####################


@gf.cell
def straight_metal(
    length: float = 10,
    cross_section: CrossSectionSpec = "metal_routing",
    width: float | None = None,
) -> gf.Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        cross_section: specification (CrossSection, string or dict).
        width: width of the waveguide. If None, it will use the width of the cross_section.
    """
    return gf.c.straight(
        length=length, cross_section=cross_section, width=width, npoints=2
    )


@gf.cell
def bend_metal(
    radius: float | None = None,
    angle: float = 90,
    width: float | None = None,
    cross_section: CrossSectionSpec = "metal_routing",
) -> gf.Component:
    """Regular degree euler bend."""
    if radius is None:
        if width:
            xs = gf.get_cross_section(cross_section=cross_section, width=width)
        else:
            xs = gf.get_cross_section(cross_section=cross_section)
        radius = xs.radius or xs.width
    return gf.c.bend_circular(
        radius=radius,
        angle=angle,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=True,
        npoints=None,
        layer=None,
    )


@gf.cell
def bend_s_metal(
    size: Size = (11, 1.8),
    cross_section: CrossSectionSpec = "metal_routing",
    width: float | None = None,
    allow_min_radius_violation: bool = True,
    npoints: int = 99,
) -> gf.Component:
    """Return S bend with bezier curve.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: in x and y direction.
        cross_section: spec.
        width: width of the waveguide. If None, it will use the width of the cross_section.
        allow_min_radius_violation: allows min radius violations.
        npoints: number of points.
    """
    return gf.c.bend_s(
        size=size,
        cross_section=cross_section,
        npoints=npoints,
        allow_min_radius_violation=allow_min_radius_violation,
        width=width,
    )
