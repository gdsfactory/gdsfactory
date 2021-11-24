""" bends with grating couplers inside the spiral
"""
from typing import Optional, Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler, bend_euler180
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.cross_section import strip
from gdsfactory.routing.manhattan import round_corners
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import ComponentFactory, CrossSectionFactory


def get_bend_port_distances(bend: Component) -> Tuple[float, float]:
    p0, p1 = bend.ports.values()
    return abs(p0.x - p1.x), abs(p0.y - p1.y)


@gf.cell
def spiral_inner_io(
    N: int = 6,
    x_straight_inner_right: float = 150.0,
    x_straight_inner_left: float = 50.0,
    y_straight_inner_top: float = 50.0,
    y_straight_inner_bottom: float = 10.0,
    grating_spacing: float = 127.0,
    waveguide_spacing: float = 3.0,
    bend90_function: ComponentFactory = bend_euler,
    bend180_function: ComponentFactory = bend_euler180,
    straight: ComponentFactory = straight_function,
    length: Optional[float] = None,
    cross_section: CrossSectionFactory = strip,
    cross_section_bend: Optional[CrossSectionFactory] = None,
    **kwargs
) -> Component:
    """Returns Spiral with ports inside the spiral loop.
    You can add grating couplers inside to save space.

    Args:
        N: number of loops
        x_straight_inner_right:
        x_straight_inner_left:
        y_straight_inner_top:
        y_straight_inner_bottom:
        grating_spacing: defaults to 127 for fiber array
        waveguide_spacing: center to center spacing
        bend90_function
        bend180_function
        straight: straight function
        length: computes spiral length from simple interpolation
        cross_section:
        cross_section_bend: for the bends
        kwargs: cross_section settings

    """
    dx = dy = waveguide_spacing
    x = cross_section(**kwargs)
    width = x.info.get("width")
    layer = x.info.get("layer")
    cross_section_bend = cross_section_bend or cross_section

    if length:
        x_straight_inner_left = get_straight_length(
            length=length,
            spiral_function=spiral_inner_io,
            N=N,
            x_straight_inner_right=x_straight_inner_right,
            x_straight_inner_left=x_straight_inner_left,
            y_straight_inner_top=y_straight_inner_top,
            y_straight_inner_bottom=y_straight_inner_bottom,
            grating_spacing=grating_spacing,
            waveguide_spacing=waveguide_spacing,
        )

    _bend180 = gf.call_if_func(
        bend180_function, cross_section=cross_section_bend, **kwargs
    )
    _bend90 = gf.call_if_func(
        bend90_function, cross_section=cross_section_bend, **kwargs
    )

    rx, ry = get_bend_port_distances(_bend90)
    _, rx180 = get_bend_port_distances(_bend180)  # rx180, second arg since we rotate

    component = Component()

    p1 = gf.Port(
        name="o1",
        midpoint=(0, y_straight_inner_top),
        orientation=270,
        width=width,
        layer=layer,
    )
    p2 = gf.Port(
        name="o2",
        midpoint=(grating_spacing, y_straight_inner_top),
        orientation=270,
        width=width,
        layer=layer,
    )

    component.add_port(name="o1", port=p1)
    component.add_port(name="o2", port=p2)

    # Create manhattan path going from west grating to westest port of bend 180
    _pt = np.array(p1.position)
    pts_w = [_pt]

    for i in range(N):
        y1 = y_straight_inner_top + ry + (2 * i + 1) * dy
        x2 = grating_spacing + 2 * rx + x_straight_inner_right + (2 * i + 1) * dx
        y3 = -y_straight_inner_bottom - ry - (2 * i + 3) * dy
        x4 = -x_straight_inner_left - (2 * i + 1) * dx
        if i == N - 1:
            x4 = x4 - rx180 + dx

        _pt1 = np.array([_pt[0], y1])
        _pt2 = np.array([x2, _pt1[1]])
        _pt3 = np.array([_pt2[0], y3])
        _pt4 = np.array([x4, _pt3[1]])
        _pt5 = np.array([_pt4[0], 0])
        _pt = _pt5

        pts_w += [_pt1, _pt2, _pt3, _pt4, _pt5]

    route_west = round_corners(
        pts_w, bend=_bend90, straight=straight, cross_section=cross_section, **kwargs
    )
    component.add(route_west.references)

    # Add loop back
    bend180_ref = _bend180.ref(port_id="o2", position=route_west.ports[1], rotation=90)
    component.add(bend180_ref)

    # Create manhattan path going from east grating to eastest port of bend 180
    _pt = np.array(p2.position)
    pts_e = [_pt]

    for i in range(N):
        y1 = y_straight_inner_top + ry + (2 * i) * dy
        x2 = grating_spacing + 2 * rx + x_straight_inner_right + 2 * i * dx
        y3 = -y_straight_inner_bottom - ry - (2 * i + 2) * dy
        x4 = -x_straight_inner_left - (2 * i) * dx

        _pt1 = np.array([_pt[0], y1])
        _pt2 = np.array([x2, _pt1[1]])
        _pt3 = np.array([_pt2[0], y3])
        _pt4 = np.array([x4, _pt3[1]])
        _pt5 = np.array([_pt4[0], 0])
        _pt = _pt5

        pts_e += [_pt1, _pt2, _pt3, _pt4, _pt5]

    route_east = round_corners(
        pts_e, bend=_bend90, straight=straight, cross_section=cross_section, **kwargs
    )
    component.add(route_east.references)

    length = route_east.length + route_west.length + _bend180.info.length
    component.info.length = snap_to_grid(length + 2 * y_straight_inner_top)
    return component


@gf.cell
def spiral_inner_io_fiber_single(
    cross_section: CrossSectionFactory = strip,
    cross_section_bend: Optional[CrossSectionFactory] = None,
    cross_section_ports: Optional[CrossSectionFactory] = None,
    x_straight_inner_right: float = 40.0,
    x_straight_inner_left: float = 75.0,
    y_straight_inner_top: float = 10.0,
    y_straight_inner_bottom: float = 0.0,
    grating_spacing: float = 200.0,
    **kwargs
):
    """Returns Spiral with 90 and 270 degree ports.
    You can add single fiber north and south grating couplers
    inside the spiral to save space

    Args:
        cross_section: for the straight sections in the spiral
        cross_section_bend: for the bends in the spiral
        cross_section_ports: for input/output ports
        x_straight_inner_right:
        x_straight_inner_left:
        y_straight_inner_top:
        y_straight_inner_bottom:
        grating_spacing:
        N: number of loops
        waveguide_spacing: center to center spacing
        bend90_function
        bend180_function
        straight: straight function
        length: computes spiral length from simple interpolation
        kwargs: cross_section settings
    """
    c = Component()
    spiral = spiral_inner_io(
        cross_section=cross_section,
        cross_section_bend=cross_section_bend,
        x_straight_inner_right=x_straight_inner_right,
        x_straight_inner_left=x_straight_inner_left,
        y_straight_inner_top=y_straight_inner_top,
        y_straight_inner_bottom=y_straight_inner_bottom,
        grating_spacing=grating_spacing,
        **kwargs
    )
    ref = c << spiral
    ref.rotate(90)

    cross_section_ports = cross_section_ports or cross_section_bend or cross_section
    bend = bend_euler(cross_section=cross_section_ports)
    btop = c << bend
    bbot = c << bend
    c.copy_child_info(spiral)

    bbot.connect("o2", ref.ports["o1"])
    btop.connect("o1", ref.ports["o2"])
    c.add_port("o2", port=btop.ports["o2"])
    c.add_port("o1", port=bbot.ports["o1"])
    return c


def get_straight_length(
    length: float, spiral_function: ComponentFactory, **kwargs
) -> float:
    """Returns y_spiral to achieve a particular spiral length"""
    x0 = 50
    x1 = 400
    kwargs.update({"x_straight_inner_left": x0})
    spiral0 = spiral_function(**kwargs)

    kwargs.update({"x_straight_inner_left": x1})
    spiral1 = spiral_function(**kwargs)
    p = np.polyfit(
        np.array([x0, x1]), np.array([spiral0.info.length, spiral1.info.length]), deg=1
    )
    x_straight_inner_left = (length - p[1]) / p[0]
    # print(x_straight_inner_left)
    return x_straight_inner_left


if __name__ == "__main__":
    # c = spiral_inner_io(radius=20, width=0.2)
    cross_section_wide = gf.partial(gf.cross_section.strip_auto_widen)

    c = spiral_inner_io_fiber_single(
        # width=2,
        # length=10e3,
        # cross_section=cross_section_wide,
        cross_section=gf.cross_section.pin,
        waveguide_spacing=30,
    )
    c = gf.add_grating_couplers.add_grating_couplers_with_loopback_fiber_single(
        c,
        loopback_xspacing=100,
    )
    c.show()
