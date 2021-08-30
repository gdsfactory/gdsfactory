""" bends with grating couplers inside the spiral
"""
from typing import Optional, Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler, bend_euler180
from gdsfactory.components.straight import straight
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
    waveguide_spacing: float = 3,
    bend90_function: ComponentFactory = bend_euler,
    bend180_function: ComponentFactory = bend_euler180,
    straight_factory: ComponentFactory = straight,
    taper: Optional[ComponentFactory] = None,
    length: Optional[float] = None,
    cross_section: CrossSectionFactory = strip,
    **kwargs
) -> Component:
    """Spiral with ports inside the spiral loop.

    Args:
        N: number of loops
        x_straight_inner_right:
        x_straight_inner_left:
        y_straight_inner_top:
        y_straight_inner_bottom:
        grating_spacing:
        waveguide_spacing: center to center spacing
        bend90_function
        bend180_function
        straight_factory: straight function
        taper: taper function
        length: computes spiral length from simple interpolation
        cross_section:
        **kwargs: cross_section settings

    """
    dx = dy = waveguide_spacing
    cross_section = gf.partial(cross_section, **kwargs)
    x = cross_section()
    width = x.info.get("width")
    taper_length = x.info.get("taper_length", 10.0)

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

    _bend180 = gf.call_if_func(bend180_function, cross_section=cross_section)
    _bend90 = gf.call_if_func(bend90_function, cross_section=cross_section)

    rx, ry = get_bend_port_distances(_bend90)
    _, rx180 = get_bend_port_distances(_bend180)  # rx180, second arg since we rotate

    component = gf.Component()
    # gc_port_lbl = 1
    # gc1 = _gc.ref(port_id=gc_port_lbl, position=(0, 0), rotation=-90)
    # gc2 = _gc.ref(port_id=gc_port_lbl, position=(grating_spacing, 0), rotation=-90)
    # component.add([gc1, gc2])

    p1 = gf.Port(
        name="o1",
        midpoint=(0, y_straight_inner_top),
        orientation=270,
        width=width,
        layer=gf.LAYER.WG,
    )
    p2 = gf.Port(
        name="o2",
        midpoint=(grating_spacing, y_straight_inner_top),
        orientation=270,
        width=width,
        layer=gf.LAYER.WG,
    )
    taper = gf.components.taper(
        width1=width,
        width2=_bend180.ports["o1"].width,
        length=taper_length + y_straight_inner_top - 15 - 35,
        cross_section=cross_section,
    )
    taper_ref1 = component.add_ref(taper)
    taper_ref1.connect("o2", p1)

    taper_ref2 = component.add_ref(taper)
    taper_ref2.connect("o2", p2)

    component.absorb(taper_ref1)
    component.absorb(taper_ref2)

    component.add_port(name="o1", port=taper_ref1.ports["o1"])
    component.add_port(name="o2", port=taper_ref2.ports["o1"])

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
        pts_w,
        bend_factory=_bend90,
        straight_factory=straight_factory,
        taper=taper,
        cross_section=cross_section,
    )
    component.add(route_west.references)

    # Add loop back
    bend180_ref = _bend180.ref(port_id="o2", position=route_west.ports[1], rotation=90)
    component.add(bend180_ref)
    component.absorb(bend180_ref)

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
        pts_e,
        bend_factory=_bend90,
        straight_factory=straight_factory,
        taper=taper,
        cross_section=cross_section,
    )
    component.add(route_east.references)

    length = route_east.length + route_west.length + _bend180.length
    component.length = snap_to_grid(length + 2 * y_straight_inner_top)
    return component


def get_straight_length(
    length: float, spiral_function: ComponentFactory, **kwargs
) -> float:
    """Returns y_spiral to achieve a particular spiral length"""
    x0 = 50
    x1 = 400
    kwargs.update({"x_straight_inner_left": x0})
    s0 = spiral_function(**kwargs)
    kwargs.update({"x_straight_inner_left": x1})
    s1 = spiral_function(**kwargs)
    p = np.polyfit(np.array([x0, x1]), np.array([s0.length, s1.length]), deg=1)
    x_straight_inner_left = (length - p[1]) / p[0]
    # print(x_straight_inner_left)
    return x_straight_inner_left


if __name__ == "__main__":
    # c = spiral_inner_io(radius=20, width=0.2)
    c = spiral_inner_io(radius=40, width=2.0, length=15e3)
    c.show()
