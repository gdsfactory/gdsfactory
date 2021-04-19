from typing import Optional

from pp.cell import cell
from pp.component import Component
from pp.cross_section import strip
from pp.path import arc, component
from pp.snap import snap_to_grid
from pp.types import CrossSectionFactory


@cell
def bend_circular(
    radius: float = 10.0,
    angle: int = 90,
    npoints: int = 720,
    snap_to_grid_nm: int = 1,
    cross_section_factory: Optional[CrossSectionFactory] = None,
    **cross_section_settings
) -> Component:
    """Returns a radial arc.

    Args:
        radius
        angle: angle of arc (degrees)
        npoints: Number of points used per 360 degrees
        snap_to_grid_nm: snaps points a nm grid
        cross_section_factory: function that returns a cross_section
        **cross_section_settings

    .. plot::
        :include-source:

        import pp

        c = pp.components.bend_circular(radius=10, angle=90, npoints=720)
        c.plot()

    """
    cross_section_factory = cross_section_factory or strip
    cross_section = cross_section_factory(**cross_section_settings)
    p = arc(radius=radius, angle=angle, npoints=npoints)
    c = component(p, cross_section, snap_to_grid_nm=snap_to_grid_nm)

    c.length = snap_to_grid(p.length())
    c.dy = abs(p.points[0][0] - p.points[-1][0])
    c.radius_min = radius
    return c


@cell
def bend_circular180(angle: int = 180, **kwargs) -> Component:
    """Returns a 180 degrees radial arc.

    Args:
        radius
        angle: angle of arc (degrees)
        npoints: number of points
        width: waveguide width (defaults to tech.wg_width)

    """
    return bend_circular(angle=angle, **kwargs)


if __name__ == "__main__":
    from pprint import pprint

    c = bend_circular()
    c.show()
    pprint(c.get_settings())
    # c.plotqt()

    # from phidl.quickplotter import quickplot2
    # c = bend_circular_trenches()
    # c = bend_circular_deep_rib()
    # print(c.ports)
    # print(c.length, np.pi * 10)
    # print(c.ports.keys())
    # print(c.ports["N0"].midpoint)
    # print(c.settings)
    # c = bend_circular_slot()
    # c = bend_circular(width=0.45, radius=5)
    # c.plot()
    # quickplot2(c)
