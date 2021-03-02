from typing import Optional

from pp.cell import cell
from pp.component import Component
from pp.path import arc, component
from pp.snap import snap_to_grid
from pp.tech import TECH_SILICON_C, Tech


@cell
def bend_circular(
    radius: float = 10.0,
    angle: int = 90,
    npoints: int = 720,
    tech: Tech = TECH_SILICON_C,
    width: Optional[float] = None,
) -> Component:
    """Returns a radial arc.

    Args:
        radius
        angle: angle of arc (degrees)
        npoints: Number of points used per 360 degrees
        width: waveguide width (defaults to tech.wg_width)
        tech: Technology

    .. plot::
      :include-source:

      import pp

      c = pp.c.bend_circular(
        radius=10,
        angle=90,
        npoints=720,
      )
      c.plot()

    """
    width = width or tech.wg_width
    cross_section = tech.get_cross_section(width=width)

    p = arc(radius=radius, angle=angle, npoints=npoints)
    c = component(p, cross_section)
    c.length = snap_to_grid(p.length())
    c.dx = abs(p.points[0][0] - p.points[-1][0])
    c.dy = abs(p.points[0][0] - p.points[-1][0])
    return c


@cell
def bend_circular180(angle=180, **kwargs) -> Component:
    """Returns an arc of length ``theta`` starting at angle ``start_angle``

    Args:
        radius
        angle: angle of arc (degrees)
        npoints: number of points
        width: waveguide width (defaults to tech.wg_width)
        tech: Technology

    """
    return bend_circular(angle=angle, **kwargs)


if __name__ == "__main__":
    from pprint import pprint

    c = bend_circular180()
    c.show()
    pprint(c.get_settings())
    # c.plotqt()

    # from phidl.quickplotter import quickplot2
    # c = bend_circular_trenches()
    # c = bend_circular_deep_rib()
    # print(c.ports)
    # c = bend_circular(radius=5.0005, width=1.002, theta=180)
    # print(c.length, np.pi * 10)
    # print(c.ports.keys())
    # print(c.ports["N0"].midpoint)
    # print(c.settings)
    # c = bend_circular_slot()
    # c = bend_circular(width=0.45, radius=5)
    # c.plot()
    # quickplot2(c)
