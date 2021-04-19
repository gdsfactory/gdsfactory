from typing import Optional

from pp.cell import cell
from pp.component import Component
from pp.cross_section import strip
from pp.path import component, euler
from pp.snap import snap_to_grid
from pp.types import CrossSectionFactory


@cell
def bend_euler(
    radius: float = 10.0,
    angle: int = 90,
    p: float = 1,
    with_arc_floorplan: bool = True,
    npoints: int = 720,
    snap_to_grid_nm: int = 1,
    cross_section_factory: Optional[CrossSectionFactory] = None,
    **cross_section_settings
) -> Component:
    """Returns an euler bend that adiabatically transitions from straight to curved.
    By default, `radius` corresponds to the minimum radius of curvature of the bend.
    However, if `use_eff` is set to True, `radius` corresponds to the effective
    radius of curvature (making the curve a drop-in replacement for an arc). If
    p < 1.0, will create a "partial euler" curve as described in Vogelbacher et.
    al. https://dx.doi.org/10.1364/oe.27.031394

    Args:
        radius: minimum radius of curvature
        angle: total angle of the curve
        p: Proportion of the curve that is an Euler curve
        with_arc_floorplan: If False: `radius` is the minimum radius of curvature of the bend
            If True: The curve will be scaled such that the endpoints match a bend_circular
            with parameters `radius` and `angle`
        npoints: Number of points used per 360 degrees
        snap_to_grid_nm: snaps points a nm grid
        cross_section_factory: function that returns a cross_section
        **cross_section_settings


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
    cross_section_factory = cross_section_factory or strip

    cross_section = cross_section_factory(**cross_section_settings)
    p = euler(
        radius=radius, angle=angle, p=p, use_eff=with_arc_floorplan, npoints=npoints
    )
    c = component(p, cross_section, snap_to_grid_nm=snap_to_grid_nm)
    c.length = snap_to_grid(p.length())
    c.dy = abs(p.points[0][0] - p.points[-1][0])
    c.radius_min = p.info["Rmin"]
    return c


@cell
def bend_euler180(angle: int = 180, **kwargs) -> Component:
    return bend_euler(angle=angle, **kwargs)


@cell
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
    cross_section = pp.cross_section.strip(width=0.5, layer=pp.LAYER.WG)

    c1 = pp.path.component(p1, cross_section=cross_section)
    c1.name = "two_90_euler"
    c2 = pp.path.component(p2, cross_section=cross_section)
    c2.name = "one_180_euler"
    c1.add_ref(c2)
    c1.show()


if __name__ == "__main__":
    c = bend_euler_s()
    c.show()

    # _compare_bend_euler180()
    # import pp
    # c = bend_euler(radius=10)
    # c << pp.components.bend_circular(radius=10)
    # c.pprint()
