import pp
from pp.add_padding import get_padding_points
from pp.component import Component
from pp.cross_section import cross_section, get_waveguide_settings
from pp.path import euler, extrude
from pp.snap import snap_to_grid


@pp.cell_with_validator
def bend_euler(
    angle: int = 90,
    p: float = 1,
    with_arc_floorplan: bool = True,
    npoints: int = 720,
    direction="ccw",
    with_cladding_box: bool = True,
    waveguide: str = "strip",
    **kwargs
) -> Component:
    """Returns an euler bend that adiabatically transitions from straight to curved.
    By default, `radius` corresponds to the minimum radius of curvature of the bend.
    However, if `use_eff` is set to True, `radius` corresponds to the effective
    radius of curvature (making the curve a drop-in replacement for an arc). If
    p < 1.0, will create a "partial euler" curve as described in Vogelbacher et.
    al. https://dx.doi.org/10.1364/oe.27.031394

    Args:
        angle: total angle of the curve
        p: Proportion of the curve that is an Euler curve
        with_arc_floorplan: If False: `radius` is the minimum radius of curvature of the bend
            If True: The curve will be scaled such that the endpoints match a bend_circular
            with parameters `radius` and `angle`
        npoints: Number of points used per 360 degrees
        direction: cw (clock-wise) or ccw (counter clock-wise)
        with_cladding_box: to avoid DRC acute angle errors in cladding
        waveguide: from tech.waveguide
        kwargs: waveguide_settings


    .. plot::
      :include-source:

      import pp

      c = pp.components.bend_euler(
        radius=10,
        angle=0.5,
        p=1,
        use_eff=False
        npoints=720,
      )
      c.plot()

    """
    waveguide_settings = get_waveguide_settings(waveguide, **kwargs)
    x = cross_section(**waveguide_settings)
    radius = x.info["radius"]
    p = euler(
        radius=radius, angle=angle, p=p, use_eff=with_arc_floorplan, npoints=npoints
    )
    c = extrude(p, x)
    c.length = snap_to_grid(p.length())
    c.dy = abs(p.points[0][0] - p.points[-1][0])
    c.radius_min = p.info["Rmin"]
    c.radius = radius
    c.layer = x.info["layer"]

    if with_cladding_box and x.info["layers_cladding"]:
        layers_cladding = x.info["layers_cladding"]
        cladding_offset = x.info["cladding_offset"]
        top = cladding_offset if angle == 180 else 0
        points = get_padding_points(
            component=c,
            default=0,
            bottom=cladding_offset,
            right=cladding_offset,
            top=top,
        )
        for layer in layers_cladding or []:
            c.add_polygon(points, layer=layer)

    if direction == "cw":
        c.mirror(p1=[0, 0], p2=[1, 0])
    return c


@pp.cell_with_validator
def bend_euler180(angle: int = 180, **kwargs) -> Component:
    return bend_euler(angle=angle, **kwargs)


@pp.cell_with_validator
def bend_euler_s(**kwargs) -> Component:
    """Sbend made of euler bends."""
    c = Component()
    b = bend_euler(**kwargs)
    b1 = c.add_ref(b)
    b2 = c.add_ref(b)
    b2.mirror()
    b2.connect("W0", b1.ports["N0"])
    c.add_port("W0", port=b1.ports["W0"])
    c.add_port("E0", port=b2.ports["N0"])
    return c


def _compare_bend_euler180():
    """Compare 180 bend euler with 2 90deg euler bends."""
    import pp

    p1 = pp.Path()
    p1.append([pp.path.euler(angle=90), pp.path.euler(angle=90)])
    p2 = pp.path.euler(angle=180)
    x = cross_section()

    c1 = pp.path.extrude(p1, x)
    c1.name = "two_90_euler"
    c2 = pp.path.extrude(p2, x)
    c2.name = "one_180_euler"
    c1.add_ref(c2)
    c1.show()


if __name__ == "__main__":
    c = bend_euler_s()
    c = bend_euler180()
    c = bend_euler(direction="cw")
    c = bend_euler()
    c.show()

    _compare_bend_euler180()
    # import pp
    # c = bend_euler(radius=10)
    # c << pp.components.bend_circular(radius=10)
    # c.pprint()
