""" bends with grating couplers inside the spiral
"""
from typing import Optional, Tuple

import numpy as np

import gdsfactory as gf
from gdsfactory.add_termination import add_gratings_and_loopback
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular, bend_circular180
from gdsfactory.components.bend_euler import bend_euler, bend_euler180
from gdsfactory.components.grating_coupler.elliptical import grating_coupler_elliptical
from gdsfactory.components.straight import straight
from gdsfactory.cross_section import get_waveguide_settings
from gdsfactory.routing.manhattan import round_corners
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import ComponentFactory, Number


def get_bend_port_distances(bend: Component) -> Tuple[float, float]:
    p0, p1 = bend.ports.values()
    return abs(p0.x - p1.x), abs(p0.y - p1.y)


@gf.cell
def spiral_inner_io(
    N: int = 6,
    x_straight_inner_right: float = 150.0,
    x_straight_inner_left: float = 150.0,
    y_straight_inner_top: float = 50.0,
    y_straight_inner_bottom: float = 10.0,
    grating_spacing: float = 127.0,
    dx: float = 3.0,
    dy: float = 3.0,
    bend90_function: ComponentFactory = bend_circular,
    bend180_function: ComponentFactory = bend_circular180,
    width: float = 0.5,
    width_grating_coupler: float = 0.5,
    straight_factory: ComponentFactory = straight,
    taper: Optional[ComponentFactory] = None,
    length: Optional[float] = None,
    waveguide: str = "strip",
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
        dx: center to center x-spacing
        dy: center to center y-spacing
        bend90_function
        bend180_function
        straight_factory: straight function
        taper: taper function
        length: Optional length (will compute spiral length)
        waveguide: waveguide definition in TECH.waveguide
        **kwargs: waveguide settings

    """
    waveguide_settings = get_waveguide_settings(waveguide, **kwargs)
    width = waveguide_settings.get("width")
    taper_length = waveguide_settings.get("taper_length", 10.0)

    if length:
        if bend180_function == bend_circular180:
            x_straight_inner_left = get_straight_length(
                length=length,
                spiral_function=spiral_inner_io,
                N=N,
                x_straight_inner_right=x_straight_inner_right,
                x_straight_inner_left=x_straight_inner_left,
                y_straight_inner_top=y_straight_inner_top,
                y_straight_inner_bottom=y_straight_inner_bottom,
                grating_spacing=grating_spacing,
                dx=dx,
                dy=dy,
                straight_factory=straight,
                bend90_function=bend_euler,
                bend180_function=bend_euler180,
            )
        else:
            x_straight_inner_left = get_straight_length(
                length=length,
                spiral_function=spiral_inner_io_euler,
                N=N,
                x_straight_inner_right=x_straight_inner_right,
                x_straight_inner_left=x_straight_inner_left,
                y_straight_inner_top=y_straight_inner_top,
                y_straight_inner_bottom=y_straight_inner_bottom,
                grating_spacing=grating_spacing,
                dx=dx,
                dy=dy,
            )

    _bend180 = gf.call_if_func(bend180_function, **waveguide_settings)
    _bend90 = gf.call_if_func(bend90_function, **waveguide_settings)

    rx, ry = get_bend_port_distances(_bend90)
    _, rx180 = get_bend_port_distances(_bend180)  # rx180, second arg since we rotate

    component = gf.Component()
    # gc_port_lbl = "W0"
    # gc1 = _gc.ref(port_id=gc_port_lbl, position=(0, 0), rotation=-90)
    # gc2 = _gc.ref(port_id=gc_port_lbl, position=(grating_spacing, 0), rotation=-90)
    # component.add([gc1, gc2])

    p1 = gf.Port(
        name="S0",
        midpoint=(0, y_straight_inner_top),
        orientation=270,
        width=width,
        layer=gf.LAYER.WG,
    )
    p2 = gf.Port(
        name="S1",
        midpoint=(grating_spacing, y_straight_inner_top),
        orientation=270,
        width=width,
        layer=gf.LAYER.WG,
    )
    taper = gf.components.taper(
        width1=width,
        width2=_bend180.ports["W0"].width,
        length=taper_length + y_straight_inner_top - 15 - 35,
        waveguide=waveguide,
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

    route_west = round_corners(
        pts_w,
        bend_factory=_bend90,
        straight_factory=straight_factory,
        taper=taper,
        waveguide=waveguide,
        **waveguide_settings
    )
    component.add(route_west.references)

    # Add loop back
    bend180_ref = _bend180.ref(port_id="W1", position=route_west.ports[1], rotation=90)
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
        waveguide=waveguide,
        **waveguide_settings
    )
    component.add(route_east.references)

    length = route_east.length + route_west.length + _bend180.length
    component.length = snap_to_grid(length + 2 * y_straight_inner_top)
    return component


@gf.cell
def spiral_inner_io_euler(
    bend90_function: ComponentFactory = bend_euler,
    bend180_function: ComponentFactory = bend_euler180,
    **kwargs
) -> Component:
    """Spiral with euler bends."""

    return spiral_inner_io(
        bend90_function=bend90_function, bend180_function=bend180_function, **kwargs
    )


@gf.cell
def spirals_nested(bend_radius: Number = 100) -> Component:
    component = gf.Component()
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


def get_straight_length(
    length: float, spiral_function: ComponentFactory, **kwargs
) -> Number:
    """Returns y_spiral to achieve a particular spiral length"""
    x0 = 50
    x1 = 400
    kwargs.update({"x_straight_inner_left": x0})
    s0 = spiral_function(**kwargs)
    kwargs.update({"x_straight_inner_left": x1})
    s1 = spiral_function(**kwargs)
    p = np.polyfit(np.array([x0, x1]), np.array([s0.length, s1.length]), deg=1)
    return (length - p[1]) / p[0]


@gf.cell
def spiral_inner_io_with_gratings(
    grating_coupler=grating_coupler_elliptical,
    spiral=spiral_inner_io_euler,
    waveguide="strip",
    **kwargs
):
    """Returns a spiral"""
    spiral = gf.call_if_func(spiral, waveguide=waveguide, **kwargs)
    grating_coupler = gf.call_if_func(grating_coupler)

    return add_gratings_and_loopback(
        component=spiral, grating_coupler=grating_coupler, waveguide=waveguide
    )


if __name__ == "__main__":

    # c = spiral_inner_io(x_straight_inner_left=800)
    # c = spiral_inner_io_euler(length=20e3)
    # c = spiral_inner_io_euler(length_spiral=20e3, width=0.4)
    # c = spiral_inner_io_euler(length_spiral=60e3, width=0.4)
    # print(c.name)
    # print(c.settings)
    # c = add_gratings_and_loopback(c)
    # c = spirals_nested()
    # c = spiral_inner_io_euler(length=20e3)

    c = spiral_inner_io_with_gratings()
    c.show(show_ports=True)

    # c = spiral_inner_io_euler(width=1)
    # from gdsfactory.routing import add_fiber_array
    # c = spiral_inner_io_euler(length_spiral=4, width=1)
    # cc = gf.routing.add_fiber_array(c)
    # print(c.length_spiral)
    # print(get_straight_length(2, spiral_inner_io_euler))
    # print(get_straight_length(4, spiral_inner_io_euler))
    # print(get_straight_length(6, spiral_inner_io_euler))
    # c = spiral_inner_io()
    # c = spiral_inner_io_euler(y_straight_inner_top=-11)
    # c = spiral_inner_io_euler(bend_radius=20, width=0.2)
    # c = spiral_inner_io_euler(bend_radius=20, width=0.2, y_straight_inner_top=200)
    # c = reticle_mockup()
    # c = spiral_inner_io()
    # c = spiral_inner_io(bend_radius=20, width=0.2)
    # c = spirals_nested()
