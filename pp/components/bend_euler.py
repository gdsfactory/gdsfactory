from typing import Optional

from pp.cell import cell
from pp.component import Component
from pp.cross_section import CrossSectionFactory, strip
from pp.path import component, euler
from pp.snap import snap_to_grid
from pp.tech import TECH_SILICON_C, Tech
from pp.types import Layer


@cell
def bend_euler(
    radius: float = 10.0,
    angle: int = 90,
    p: float = 1,
    with_arc_floorplan: bool = True,
    npoints: int = 720,
    width: float = TECH_SILICON_C.wg_width,
    layer: Layer = TECH_SILICON_C.layer_wg,
    cross_section_factory: Optional[CrossSectionFactory] = None,
    tech: Optional[Tech] = None,
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
            If True: The curve will be scaled such that the endpoints match an arc
            with parameters `radius` and `angle`
        npoints: Number of points used per 360 degrees
        width: waveguide width (defaults to tech.wg_width)
        layer: layer for bend (defaults to tech.layer_wg)
        tech: Technology with default values
        cross_section_factory: function that returns a cross_section


    .. plot::
      :include-source:

      import pp

      c = pp.c.bend_euler(
        radius=10,
        angle=0.5,
        p=1,
        use_eff=False
        npoints=720,
      )
      c.plot()

    """
    cross_section_factory = cross_section_factory or strip
    tech = tech or TECH_SILICON_C

    cross_section = cross_section_factory(
        width=width,
        layer=layer,
        layers_cladding=tech.layers_cladding,
        cladding_offset=tech.cladding_offset,
    )
    p = euler(
        radius=radius, angle=angle, p=p, use_eff=with_arc_floorplan, npoints=npoints
    )
    c = component(p, cross_section, snap_to_grid_nm=tech.snap_to_grid_nm)
    c.length = snap_to_grid(p.length())
    c.dy = abs(p.points[0][0] - p.points[-1][0])
    c.radius_min = p.info["Rmin"]
    return c


@cell
def bend_euler180(angle: int = 180, **kwargs) -> Component:
    return bend_euler(angle=angle, **kwargs)


if __name__ == "__main__":
    c = bend_euler(radius=10)
    c.pprint()
    c.show()
