import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.component import Component
from gdsfactory.components.straight import straight
from gdsfactory.components.wire import wire_corner
from gdsfactory.cross_section import strip
from gdsfactory.path import euler, extrude
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import CrossSectionSpec


@gf.cell
def bend_euler(
    angle: float = 90.0,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int = 720,
    direction: str = "ccw",
    with_bbox: bool = True,
    cross_section: CrossSectionSpec = "strip",
    **kwargs
) -> Component:
    """Returns an euler bend that transitions from straight to curved.

    By default, `radius` corresponds to the minimum radius of curvature of the bend.
    However, if `with_arc_floorplan` is True, `radius` corresponds to the effective
    radius of curvature (making the curve a drop-in replacement for an arc). If
    p < 1.0, will create a "partial euler" curve as described in Vogelbacher et.
    al. https://dx.doi.org/10.1364/oe.27.031394

    default p = 0.5 based on this paper
    https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-25-8-9150&id=362937

    Args:
        angle: total angle of the curve.
        p: Proportion of the curve that is an Euler curve.
        with_arc_floorplan: If False: `radius` is the minimum radius of curvature
          If True: The curve scales such that the endpoints match a bend_circular
          with parameters `radius` and `angle`.
        npoints: Number of points used per 360 degrees.
        direction: cw (clock-wise) or ccw (counter clock-wise).
        with_bbox: add bbox_layers and bbox_offsets to avoid DRC sharp edges.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        kwargs: cross_section settings.

    .. code::

                  o2
                  |
                 /
                /
               /
       o1_____/
    """
    x = gf.get_cross_section(cross_section, **kwargs)
    radius = x.radius

    if radius is None:
        return wire_corner(cross_section=x)

    c = Component()
    p = euler(
        radius=radius, angle=angle, p=p, use_eff=with_arc_floorplan, npoints=npoints
    )
    ref = c << extrude(p, x)
    c.add_ports(ref.ports)
    c.info["length"] = snap_to_grid(p.length())
    c.info["dy"] = snap_to_grid(abs(float(p.points[0][0] - p.points[-1][0])))
    c.info["radius_min"] = snap_to_grid(p.info["Rmin"])
    c.info["radius"] = radius
    c.info["width"] = x.width

    if x.info:
        c.info.update(x.info)

    if with_bbox:
        padding = []
        for offset in x.bbox_offsets:
            top = offset if angle == 180 else 0
            points = get_padding_points(
                component=c,
                default=0,
                bottom=offset,
                right=offset,
                top=top,
            )
            padding.append(points)

        for layer, points in zip(x.bbox_layers, padding):
            c.add_polygon(points, layer=layer)

    if direction == "cw":
        ref.mirror(p1=[0, 0], p2=[1, 0])

    c.absorb(ref)
    return c


bend_euler180 = gf.partial(bend_euler, angle=180)


@gf.cell
def bend_euler_s(**kwargs) -> Component:
    """Sbend made of euler bends."""
    c = Component()
    b = bend_euler(**kwargs)
    b1 = c.add_ref(b)
    b2 = c.add_ref(b)
    b2.mirror()
    b2.connect("o1", b1.ports["o2"])
    c.add_port("o1", port=b1.ports["o1"])
    c.add_port("o2", port=b2.ports["o2"])
    return c


@gf.cell
def bend_straight_bend(
    straight_length: float = 10.0,
    angle: float = 90,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int = 720,
    direction: str = "ccw",
    cross_section: CrossSectionSpec = strip,
    **kwargs
) -> Component:
    """Sbend made of 2 euler bends and straight section in between.

    Args:
        straight_length: in um.
        angle: total angle of the curve.
        p: Proportion of the curve that is an Euler curve.
        with_arc_floorplan: If False: `radius` is the minimum radius of curvature
          If True: The curve scales such that the endpoints match a bend_circular
          with parameters `radius` and `angle`.
        npoints: Number of points used per 360 degrees.
        direction: cw (clock-wise) or ccw (counter clock-wise).
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        kwargs: cross_section settings.
    """
    c = Component()
    b = bend_euler(
        angle=angle,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        direction=direction,
        cross_section=cross_section,
        **kwargs
    )
    b1 = c.add_ref(b)
    b2 = c.add_ref(b)
    s = c << straight(length=straight_length, cross_section=cross_section, **kwargs)
    s.connect("o1", b1.ports["o2"])
    b2.mirror()
    b2.connect("o1", s.ports["o2"])
    c.add_port("o1", port=b1.ports["o1"])
    c.add_port("o2", port=b2.ports["o2"])
    return c


def _compare_bend_euler180() -> None:
    """Compare 180 bend euler with 2 90deg euler bends."""
    import gdsfactory as gf

    p1 = gf.Path()
    p1.append([gf.path.euler(angle=90), gf.path.euler(angle=90)])
    p2 = gf.path.euler(angle=180)
    x = gf.cross_section.strip()

    c1 = gf.path.extrude(p1, x)
    c1.name = "two_90_euler"
    c2 = gf.path.extrude(p2, x)
    c2.name = "one_180_euler"
    c1.add_ref(c2)
    c1.show()


def _compare_bend_euler90():
    """Compare bend euler with 90deg circular bend."""
    import gdsfactory as gf

    c = gf.Component()
    radius = 10
    b1 = bend_euler(radius=radius)
    b2 = gf.components.bend_circular(radius=radius)

    print(b1.info["length"])
    print(b2.info["length"])

    c << b1
    c << b2
    return c


if __name__ == "__main__":
    # c = bend_euler_s()
    c = bend_euler()
    # c = bend_euler(bbox_layers=[(2, 0), (3, 0)], bbox_offsets=[3, 3])
    c.show(show_ports=True)

    # c = bend_euler(direction="cw")
    # c = bend_euler(angle=270)
    # c.pprint()
    # p = euler()
    # c = bend_straight_bend()
    # c = _compare_bend_euler90()

    # c = gf.Component()
    # b1 = c << bend_euler()
    # b2 = c << bend_euler()
    # b2.connect("o1", b1.ports["o2"])
    # c.show(show_ports=False)

    # _compare_bend_euler180()
    # import gdsfactory as gf
    # c = bend_euler(radius=10)
    # c << gf.components.bend_circular(radius=10)
    # c.pprint()
