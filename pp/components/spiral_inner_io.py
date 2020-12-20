""" bends with grating couplers inside the spiral
maybe: need to add grating coupler loopback as well
"""

from typing import Callable, Optional, Tuple

import numpy as np

import pp
from pp.component import Component
from pp.components.bend_circular import bend_circular, bend_circular180
from pp.components.euler.bend_euler import bend_euler90, bend_euler180
from pp.components.waveguide import waveguide
from pp.config import TAPER_LENGTH
from pp.routing import round_corners


def get_bend_port_distances(bend: Component) -> Tuple[float, float]:
    p0, p1 = bend.ports.values()
    return abs(p0.x - p1.x), abs(p0.y - p1.y)


@pp.cell
def spiral_inner_io(
    N: int = 6,
    x_straight_inner_right: float = 150.0,
    x_straight_inner_left: float = 150.0,
    y_straight_inner_top: float = 50.0,
    y_straight_inner_bottom: float = 10.0,
    grating_spacing: float = 127.0,
    dx: float = 3.0,
    dy: float = 3.0,
    bend90_function: Callable = bend_circular,
    bend180_function: Callable = bend_circular180,
    bend_radius: float = 50.0,
    wg_width: float = 0.5,
    wg_width_grating_coupler: float = 0.5,
    straight_factory: Callable = waveguide,
    taper: Optional[Callable] = None,
    length: Optional[float] = None,
) -> Component:
    """Spiral with grating couplers inside.

    Args:
        N: number of loops
        x_straight_inner_right:
        x_straight_inner_left:
        y_straight_inner_top:
        y_straight_inner_bottom:
        grating_spacing:
        dx: center to center x-spacing
        dy: center to center y-spacing
        bend90_function
        bend180_function
        bend_radius
        wg_width
        straight_factory
        taper:
        length: cm

    .. plot::
      :include-source:

      import pp
      from pp.components.spiral_inner_io import spiral_inner_io

      c = spiral_inner_io()
      pp.plotgds(c)
    """
    if length:
        if bend180_function == bend_circular180:
            y_straight_inner_top = get_straight_length(
                length_cm=length,
                spiral_function=spiral_inner_io,
                N=N,
                x_straight_inner_right=x_straight_inner_right,
                x_straight_inner_left=x_straight_inner_left,
                y_straight_inner_top=y_straight_inner_top,
                y_straight_inner_bottom=y_straight_inner_bottom,
                grating_spacing=grating_spacing,
                dx=dx,
                dy=dy,
                straight_factory=waveguide,
                bend90_function=bend_euler90,
                bend180_function=bend_euler180,
                wg_width=wg_width,
            )
        else:
            y_straight_inner_top = get_straight_length(
                length_cm=length,
                spiral_function=spiral_inner_io_euler,
                N=N,
                x_straight_inner_right=x_straight_inner_right,
                x_straight_inner_left=x_straight_inner_left,
                y_straight_inner_top=y_straight_inner_top,
                y_straight_inner_bottom=y_straight_inner_bottom,
                grating_spacing=grating_spacing,
                dx=dx,
                dy=dy,
                wg_width=wg_width,
            )

    _bend180 = pp.call_if_func(bend180_function, radius=bend_radius, width=wg_width)
    _bend90 = pp.call_if_func(bend90_function, radius=bend_radius, width=wg_width)

    rx, ry = get_bend_port_distances(_bend90)
    _, rx180 = get_bend_port_distances(_bend180)  # rx180, second arg since we rotate

    component = pp.Component()
    # gc_port_lbl = "W0"
    # gc1 = _gc.ref(port_id=gc_port_lbl, position=(0, 0), rotation=-90)
    # gc2 = _gc.ref(port_id=gc_port_lbl, position=(grating_spacing, 0), rotation=-90)
    # component.add([gc1, gc2])

    p1 = pp.Port(
        name="S0",
        midpoint=(0, y_straight_inner_top),
        orientation=270,
        width=wg_width,
        layer=pp.LAYER.WG,
    )
    p2 = pp.Port(
        name="S1",
        midpoint=(grating_spacing, y_straight_inner_top),
        orientation=270,
        width=wg_width,
        layer=pp.LAYER.WG,
    )
    taper = pp.c.taper(
        width1=wg_width_grating_coupler,
        width2=_bend180.ports["W0"].width,
        length=TAPER_LENGTH + y_straight_inner_top - 15 - 35,
    )
    taper_ref1 = component.add_ref(taper)
    taper_ref1.connect("2", p1)

    taper_ref2 = component.add_ref(taper)
    taper_ref2.connect("2", p2)

    component.absorb(taper_ref1)
    component.absorb(taper_ref2)

    component.add_port(name="S0", port=taper_ref1.ports["1"])
    component.add_port(name="S1", port=taper_ref2.ports["1"])

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

    route_ref_w = round_corners(
        pts_w, bend90=_bend90, straight_factory=straight_factory, taper=taper
    )
    component.add(route_ref_w)
    component.absorb(route_ref_w)

    # Add loop back
    bend180_ref = _bend180.ref(
        port_id="W1", position=route_ref_w.ports["output"], rotation=90
    )
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

    route_ref_e = round_corners(
        pts_e, bend90=_bend90, straight_factory=straight_factory, taper=taper
    )
    component.add(route_ref_e)
    component.absorb(route_ref_e)

    length = (
        route_ref_e.info["length"]
        + route_ref_w.info["length"]
        + bend180_ref.info["length"]
    )
    component.length = pp.drc.snap_to_1nm_grid(length + 2 * y_straight_inner_top)
    return component


@pp.cell
def spiral_inner_io_euler(
    bend90_function: Callable = bend_euler90,
    bend180_function: Callable = bend_euler180,
    **kwargs
) -> Component:
    """

    .. plot::
      :include-source:

      import pp

      c = pp.c.spiral_inner_io_euler()
      pp.plotgds(c)
    """

    return spiral_inner_io(
        bend90_function=bend90_function, bend180_function=bend180_function, **kwargs
    )


@pp.cell
def spirals_nested(bend_radius=100):
    component = pp.Component()
    c = spiral_inner_io(
        N=42,
        y_straight_inner_top=10.0,
        y_straight_inner_bottom=5700.0,
        x_straight_inner_right=2000.0,
        x_straight_inner_left=20.0,
        bend_radius=bend_radius,
    )

    c1 = spiral_inner_io(
        N=42,
        y_straight_inner_top=10.0,
        y_straight_inner_bottom=10.0,
        x_straight_inner_right=0.0,
        x_straight_inner_left=120.0,
        bend_radius=bend_radius,
    )

    c2 = spiral_inner_io(
        N=42,
        y_straight_inner_top=10.0,
        y_straight_inner_bottom=2000.0,
        x_straight_inner_right=0.0,
        x_straight_inner_left=120.0,
        bend_radius=bend_radius,
    )

    # for _c in [c, c1, c2]:
    #     print(_c.info["length"])

    component.add(c.ref(position=(0, 0)))
    component.add(c1.ref(position=(1150, -850)))
    component.add(c2.ref(position=(1150, -2850)))
    return component


@pp.cell
def mini_block_mockup():
    component = pp.Component()

    c = spirals_nested()
    x0 = c.size_info.west
    y0 = c.size_info.south
    margin = 50.0
    component.add(c.ref(position=(-x0 + margin, -y0 + margin)))
    component.add(c.ref(position=(4500 - x0 + margin, -y0 + margin)))

    return component


@pp.cell
def reticle_mockup():
    from pp.layers import LAYER

    dx = 9000.0
    dy = 8000.0
    a0 = 50.0
    component = pp.Component()
    for x in [0, dx, 2 * dx, 3 * dx]:
        a = x + a0
        b = x - a0

        component.add_polygon(
            [(a, 0), (b, 0), (b, 3 * dy), (a, 3 * dy)], LAYER.FLOORPLAN
        )

    for y in [0, dy, 2 * dy, 3 * dy]:
        a = y + a0
        b = y - a0
        component.add_polygon(
            [(0, a), (0, b), (3 * dx, b), (3 * dx, a)], LAYER.FLOORPLAN
        )

    c = mini_block_mockup()

    for i, j in [(0, 0), (2, 2), (0, 2), (2, 0), (1, 1)]:
        component.add(c.ref(position=(i * dx, j * dy)))

    return component


def get_straight_length(length_cm, spiral_function, **kwargs):
    """ returns y_spiral to achieve a particular spiral length """
    y0 = 50
    y1 = 400
    kwargs.update({"y_straight_inner_top": y0})
    s0 = spiral_function(**kwargs)
    kwargs.update({"y_straight_inner_top": y1})
    s1 = spiral_function(**kwargs)
    p = np.polyfit(
        np.array([y0, y1]), 1e-6 * 1e2 * np.array([s0.length, s1.length]), deg=1
    )
    return (length_cm - p[1]) / p[0]


# @pp.cell
# def spiral_inner_io_with_gratings(
#     spiral=spiral_inner_io, grating_coupler=pp.c.grating_coupler_elliptical_te, **kwargs
# ):
#     spiral = pp.call_if_func(spiral, **kwargs)
#     grating_coupler = pp.call_if_func(grating_coupler)

#     return add_gratings_and_loop_back(spiral, grating_coupler=grating_coupler)


if __name__ == "__main__":
    from pp.add_termination import add_gratings_and_loop_back

    c = spiral_inner_io()
    # c = spiral_inner_io_euler()
    # c = spiral_inner_io_euler(length=2, wg_width=0.4)
    # c = spiral_inner_io_euler(length=6, wg_width=0.4)
    print(c.name)
    print(c.settings)
    cc = add_gratings_and_loop_back(c)
    pp.show(c)

    # c = spiral_inner_io_euler(wg_width=1)
    # from pp.routing import add_fiber_array
    # c = spiral_inner_io_euler(length=4, wg_width=1)
    # cc = pp.routing.add_fiber_array(c)
    # print(c.length)
    # print(get_straight_length(2, spiral_inner_io_euler))
    # print(get_straight_length(4, spiral_inner_io_euler))
    # print(get_straight_length(6, spiral_inner_io_euler))
    # c = spiral_inner_io()
    # c = spiral_inner_io_euler(y_straight_inner_top=-11)
    # c = spiral_inner_io_euler(bend_radius=20, wg_width=0.2)
    # c = spiral_inner_io_euler(bend_radius=20, wg_width=0.2, y_straight_inner_top=200)
    # c = reticle_mockup()
    # c = spiral_inner_io()
    # c = spiral_inner_io(bend_radius=20, wg_width=0.2)
    # c = spirals_nested()
    pp.show(c)
