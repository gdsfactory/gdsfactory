"""adapted from phidl.routing
temporary solution until we add Sbend routing functionality
"""
from typing import Optional

import gdspy
import numpy as np
from numpy import cos, mod, pi, sin
from numpy.linalg import norm

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.config import TECH
from gdsfactory.snap import snap_to_grid
from gdsfactory.types import Layer, Port, Route


class RoutingError(ValueError):
    pass


@cell
def route_basic(
    port1: Port,
    port2: Port,
    path_type: str = "sine",
    width_type: str = "straight",
    width1: Optional[float] = None,
    width2: Optional[float] = None,
    num_path_pts: int = 99,
    layer: Optional[Layer] = None,
) -> Component:
    layer = layer or port1.layer

    # Assuming they're both Ports for now
    point_a = np.array(port1.midpoint)
    if width1 is None:
        width1 = port1.width
    point_b = np.array(port2.midpoint)
    if width2 is None:
        width2 = port2.width
    if round(abs(mod(port1.orientation - port2.orientation, 360)), 3) != 180:
        raise RoutingError(
            "Route() error: Ports do not face each other (orientations must be 180 apart)"
        )
    orientation = port1.orientation

    separation = point_b - point_a  # Vector drawn from A to B
    distance = norm(separation)  # Magnitude of vector from A to B
    rotation = (
        np.arctan2(separation[1], separation[0]) * 180 / pi
    )  # Rotation of vector from A to B
    # If looking out along the normal of ``a``, the angle you would have to
    # look to see ``b``
    angle = rotation - orientation
    forward_distance = distance * cos(angle * pi / 180)
    lateral_distance = distance * sin(angle * pi / 180)

    # Create a path assuming starting at the origin and setting orientation = 0
    # use the "connect" function later to move the path to the correct location
    xf = forward_distance
    yf = lateral_distance

    def curve_fun_straight(t):
        return [xf * t, yf * t]

    def curve_deriv_fun_straight(t):
        return [xf + t * 0, t * 0]

    def curve_fun_sine(t):
        return [xf * t, yf * (1 - cos(t * pi)) / 2]

    def curve_deriv_fun_sine(t):
        return [xf + t * 0, yf * (sin(t * pi) * pi) / 2]

    def width_fun_straight(t):
        return (width2 - width1) * t + width1

    def width_fun_sine(t):
        return (width2 - width1) * (1 - cos(t * pi)) / 2 + width1

    if path_type == "straight":
        curve_fun = curve_fun_straight
        curve_deriv_fun = curve_deriv_fun_straight

    if path_type == "sine":
        curve_fun = curve_fun_sine
        curve_deriv_fun = curve_deriv_fun_sine

    # if path_type == 'semicircle':
    #    def semicircle(t):
    #        t = np.array(t)
    #        x,y = np.zeros(t.shape), np.zeros(t.shape)
    #        ii = (0 <= t) & (t <= 0.5)
    #        jj = (0.5 < t) & (t <= 1)
    #        x[ii] = (cos(-pi/2 + t[ii]*pi/2))*xf
    #        y[ii] = (sin(-pi/2 + t[ii]*pi/2)+1)*yf*2
    #        x[jj] = (cos(pi*3/2 - t[jj]*pi)+2)*xf/2
    #        y[jj] = (sin(pi*3/2 - t[jj]*pi)+1)*yf/2
    #        return x,y
    #    curve_fun = semicircle
    #    curve_deriv_fun = None
    if width_type == "straight":
        width_fun = width_fun_straight

    if width_type == "sine":
        width_fun = width_fun_sine

    route_path = gdspy.Path(width=width1, initial_point=(0, 0))
    route_path.parametric(
        curve_fun,
        curve_deriv_fun,
        number_of_evaluations=num_path_pts,
        max_points=199,
        final_width=width_fun,
        final_distance=None,
    )
    route_path_polygons = route_path.polygons

    # Make the route path into a Device with ports, and use "connect" to move it
    # into the proper location
    D = Component()
    D.add_polygon(route_path_polygons, layer=layer)
    p1 = D.add_port(name=1, midpoint=(0, 0), width=width1, orientation=180)
    D.add_port(
        name=2,
        midpoint=[forward_distance, lateral_distance],
        width=width2,
        orientation=0,
    )
    D.info["length"] = route_path.length

    D.rotate(angle=180 + port1.orientation - p1.orientation, center=p1.midpoint)
    D.move(origin=p1, destination=port1)
    return D


@cell
def _arc(
    radius=10, width=0.5, theta=90, start_angle=0, angle_resolution=2.5, layer=(1, 0)
):
    """Creates an arc of arclength ``theta`` starting at angle ``start_angle``"""
    inner_radius = radius - width / 2
    outer_radius = radius + width / 2
    angle1 = (start_angle) * pi / 180
    angle2 = (start_angle + theta) * pi / 180
    t = np.linspace(angle1, angle2, int(np.ceil(abs(theta) / angle_resolution)))
    inner_points_x = (inner_radius * cos(t)).tolist()
    inner_points_y = (inner_radius * sin(t)).tolist()
    outer_points_x = (outer_radius * cos(t)).tolist()
    outer_points_y = (outer_radius * sin(t)).tolist()
    xpts = inner_points_x + outer_points_x[::-1]
    ypts = inner_points_y + outer_points_y[::-1]

    D = Component()
    D.add_polygon(points=(xpts, ypts), layer=layer)
    D.add_port(
        name=1,
        midpoint=(radius * cos(angle1), radius * sin(angle1)),
        width=width,
        orientation=start_angle - 90 + 180 * (theta < 0),
    )
    D.add_port(
        name=2,
        midpoint=(radius * cos(angle2), radius * sin(angle2)),
        width=width,
        orientation=start_angle + theta + 90 - 180 * (theta < 0),
    )
    D.info["length"] = (abs(theta) * pi / 180) * radius
    return D


@cell
def _gradual_bend(
    radius=10,
    width=1.0,
    angular_coverage=15,
    num_steps=10,
    angle_resolution=0.1,
    start_angle=0,
    direction="ccw",
    layer=0,
):
    """
    creates a 90-degree bent waveguide
    the bending radius is gradually increased until it reaches the minimum
    value of the radius at the "angular coverage" angle.
    it essentially creates a smooth transition to a bent waveguide mode.
    user can control number of steps provided.
    direction determined by start angle and cw or ccw switch
    ############
    with the default 10 "num_steps" and 15 degree coverage, effective radius is about 1.5*radius.
    """
    angular_coverage = np.deg2rad(angular_coverage)
    D = Component()

    # determines the increment in radius through its inverse from 0 to 1/r
    inc_rad = (radius ** -1) / (num_steps)
    angle_step = angular_coverage / num_steps

    # construct a series of sub-arcs with equal angles but gradually
    # decreasing bend radius
    arcs = []
    prevPort = None
    for x in range(num_steps):
        A = _arc(
            radius=1 / ((x + 1) * inc_rad),
            width=width,
            theta=np.rad2deg(angle_step),
            start_angle=x * np.rad2deg(angle_step),
            angle_resolution=angle_resolution,
            layer=layer,
        )
        a = D.add_ref(A)

        arcs.append(a)
        if x > 0:
            a.connect(port=1, destination=prevPort)
        prevPort = a.ports[2]
        D.absorb(a)
    D.add_port(name=1, port=arcs[0].ports[1])

    # now connect a regular bend for the normal curved portion
    B = _arc(
        radius=radius,
        width=width,
        theta=45 - np.rad2deg(angular_coverage),
        start_angle=angular_coverage,
        angle_resolution=angle_resolution,
        layer=layer,
    )
    b = D.add_ref(B)
    b.connect(port=1, destination=prevPort)
    prevPort = b.ports[2]
    D.add_port(name=2, port=prevPort)

    # now create the overall structure
    Total = Component()

    # clone the half-curve into two objects and connect for a 90 deg bend.
    D1 = Total.add_ref(D)
    D2 = Total.add_ref(D)
    D2.mirror(p1=[0, 0], p2=[1, 1])
    D2.connect(port=2, destination=D1.ports[2])
    Total.xmin = 0
    Total.ymin = 0

    # orient to default settings...
    Total.mirror(p1=[0, 0], p2=[1, 1])
    Total.mirror(p1=[0, 0], p2=[1, 0])

    # orient to user-provided settings
    if direction == "cw":
        Total.mirror(p1=[0, 0], p2=[1, 0])
    Total.rotate(angle=start_angle, center=Total.center)
    Total.center = [0, 0]
    Total.add_port(name=1, port=D1.ports[1])
    Total.add_port(name=2, port=D2.ports[1])

    Total.info["length"] = (abs(angular_coverage) * pi / 180) * radius
    Total.absorb(D1)
    Total.absorb(D2)
    return Total


def _route_manhattan180(port1, port2, bendType="circular", layer=0, radius=20):
    # this is a subroutine of route_manhattan() and should not be used by itself.
    Total = Component()
    width = port1.width
    # first map into uniform plane with normal x,y coords
    # allows each situation to be put into uniform cases of quadrants for routing.
    # this is because bends change direction and positioning.
    if port1.orientation == 0:
        p2 = [port2.midpoint[0], port2.midpoint[1]]
        p1 = [port1.midpoint[0], port1.midpoint[1]]
    if port1.orientation == 90:
        p2 = [port2.midpoint[1], -port2.midpoint[0]]
        p1 = [port1.midpoint[1], -port1.midpoint[0]]
    if port1.orientation == 180:
        p2 = [-port2.midpoint[0], -port2.midpoint[1]]
        p1 = [-port1.midpoint[0], -port1.midpoint[1]]
    if port1.orientation == 270:
        p2 = [-port2.midpoint[1], port2.midpoint[0]]
        p1 = [-port1.midpoint[1], port1.midpoint[0]]

    # create placeholder ports based on the imaginary coordinates we created
    Total.add_port(name="t1", midpoint=[0, 0], orientation=0, width=width)
    if port1.orientation != port2.orientation:
        Total.add_port(
            name="t2", midpoint=list(np.subtract(p2, p1)), orientation=180, width=width
        )
    else:
        Total.add_port(
            name="t2", midpoint=list(np.subtract(p2, p1)), orientation=0, width=width
        )

    if port1.orientation == port2.orientation:
        # first quadrant target
        if (p2[1] > p1[1]) & (p2[0] > p1[0]):
            if bendType == "circular":
                B1 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=0,
                    theta=90,
                )
                B2 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=90,
                    theta=90,
                )
                radiusEff = radius
            if bendType == "gradual":
                B1 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=0,
                    direction="ccw",
                )
                B2 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=90,
                    direction="ccw",
                )
                radiusEff = B1.xsize - width / 2
            b1 = Total.add_ref(B1)
            b2 = Total.add_ref(B2)

            b1.connect(port=b1.ports[1], destination=Total.ports["t1"])
            b1.move([p2[0] - p1[0], 0])
            b2.connect(port=b2.ports[1], destination=b1.ports[2])
            b2.move([0, p2[1] - p1[1] - radiusEff * 2])
            R1 = route_basic(port1=Total.ports["t1"], port2=b1.ports[1], layer=layer)
            r1 = Total.add_ref(R1)
            R2 = route_basic(port1=b1.ports[2], port2=b2.ports[1], layer=layer)
            r2 = Total.add_ref(R2)
            Total.add_port(name=1, port=r1.ports[1])
            Total.add_port(name=2, port=b2.ports[2])
        # second quadrant target
        if (p2[1] > p1[1]) & (p2[0] < p1[0]):
            if bendType == "circular":
                B1 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=0,
                    theta=90,
                )
                B2 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=90,
                    theta=90,
                )
                radiusEff = radius
            if bendType == "gradual":
                B1 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=0,
                    direction="ccw",
                )
                B2 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=90,
                    direction="ccw",
                )
                radiusEff = B1.xsize - width / 2
            b1 = Total.add_ref(B1)
            b2 = Total.add_ref(B2)
            b1.connect(port=b1.ports[1], destination=Total.ports["t1"])

            b2.connect(port=b2.ports[1], destination=b1.ports[2])
            b2.move([0, p2[1] - p1[1] - radiusEff * 2])
            R1 = route_basic(port1=b1.ports[2], port2=b2.ports[1], layer=layer)
            r1 = Total.add_ref(R1)
            R2 = route_basic(port1=b2.ports[2], port2=Total.ports["t2"], layer=layer)
            r2 = Total.add_ref(R2)
            Total.add_port(name=1, port=b1.ports[1])
            Total.add_port(name=2, port=r2.ports[2])
        # third quadrant target
        if (p2[1] < p1[1]) & (p2[0] < p1[0]):
            if bendType == "circular":
                B1 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=0,
                    theta=-90,
                )
                B2 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=-90,
                    theta=-90,
                )
                radiusEff = radius
            if bendType == "gradual":
                B1 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=0,
                    direction="cw",
                )
                B2 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=-90,
                    direction="cw",
                )
                radiusEff = B1.xsize - width / 2
            b1 = Total.add_ref(B1)
            b2 = Total.add_ref(B2)
            b1.connect(port=b1.ports[1], destination=Total.ports["t1"])

            b2.connect(port=b2.ports[1], destination=b1.ports[2])
            b2.move([0, p2[1] - p1[1] + radiusEff * 2])
            R1 = route_basic(port1=b1.ports[2], port2=b2.ports[1], layer=layer)
            r1 = Total.add_ref(R1)
            R2 = route_basic(port1=b2.ports[2], port2=Total.ports["t2"], layer=layer)
            r2 = Total.add_ref(R2)
            Total.add_port(name=1, port=b1.ports[1])
            Total.add_port(name=2, port=r2.ports[2])
        # fourth quadrant target
        if (p2[1] < p1[1]) & (p2[0] > p1[0]):
            if bendType == "circular":
                B1 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=0,
                    theta=-90,
                )
                B2 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=-90,
                    theta=-90,
                )
                radiusEff = radius
            if bendType == "gradual":
                B1 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=0,
                    direction="cw",
                )
                B2 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=-90,
                    direction="cw",
                )
                radiusEff = B1.xsize - width / 2
            b1 = Total.add_ref(B1)
            b2 = Total.add_ref(B2)

            b1.connect(port=b1.ports[1], destination=Total.ports["t1"])
            b1.move([p2[0] - p1[0], 0])
            b2.connect(port=b2.ports[1], destination=b1.ports[2])
            b2.move([0, p2[1] - p1[1] + radiusEff * 2])
            R1 = route_basic(port1=Total.ports["t1"], port2=b1.ports[1], layer=layer)
            r1 = Total.add_ref(R1)
            R2 = route_basic(port1=b1.ports[2], port2=b2.ports[1], layer=layer)
            r2 = Total.add_ref(R2)
            Total.add_port(name=1, port=r1.ports[1])
            Total.add_port(name=2, port=b2.ports[2])

    # other port orientations are not supported:
    elif np.round(np.abs(np.mod(port1.orientation - port2.orientation, 360)), 3) != 180:
        raise ValueError(
            "Route() error: Ports do not face each other (orientations must be 180 apart)"
        )
    # otherwise, they are 180 degrees apart:
    else:
        # first quadrant target
        if (p2[1] > p1[1]) & (p2[0] > p1[0]):
            if bendType == "circular":
                B1 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=0,
                    theta=90,
                )
                B2 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=90,
                    theta=-90,
                )
                radiusEff = radius
            if bendType == "gradual":
                B1 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=0,
                    direction="ccw",
                )
                B2 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=90,
                    direction="cw",
                )
                radiusEff = B1.xsize - width / 2
            b1 = Total.add_ref(B1)
            b2 = Total.add_ref(B2)

            b1.connect(port=b1.ports[1], destination=Total.ports["t1"])
            b1.move([p2[0] - p1[0] - radiusEff * 2, 0])
            b2.connect(port=b2.ports[1], destination=b1.ports[2])
            b2.move([0, p2[1] - p1[1] - radiusEff * 2])
            R1 = route_basic(port1=Total.ports["t1"], port2=b1.ports[1], layer=layer)
            r1 = Total.add_ref(R1)
            R2 = route_basic(port1=b1.ports[2], port2=b2.ports[1], layer=layer)
            r2 = Total.add_ref(R2)
            Total.add_port(name=1, port=r1.ports[1])
            Total.add_port(name=2, port=b2.ports[2])
        # second quadrant target
        if (p2[1] > p1[1]) & (p2[0] < p1[0]):
            if bendType == "circular":
                B1 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=0,
                    theta=90,
                )
                B2 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=90,
                    theta=90,
                )
                B3 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=180,
                    theta=-90,
                )
                B4 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=90,
                    theta=-90,
                )
                radiusEff = radius
            if bendType == "gradual":
                B1 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=0,
                    direction="ccw",
                )
                B2 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=90,
                    direction="ccw",
                )
                B3 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=180,
                    direction="cw",
                )
                B4 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=90,
                    direction="cw",
                )
                radiusEff = B1.xsize - width / 2
            b1 = Total.add_ref(B1)
            b2 = Total.add_ref(B2)
            b3 = Total.add_ref(B3)
            b4 = Total.add_ref(B4)

            b1.connect(port=b1.ports[1], destination=Total.ports["t1"])

            b2.connect(port=b2.ports[1], destination=b1.ports[2])
            b2.move([0, p2[1] - p1[1] - radiusEff * 4])
            R1 = route_basic(port1=b1.ports[2], port2=b2.ports[1], layer=layer)
            r1 = Total.add_ref(R1)
            b3.connect(port=b3.ports[1], destination=b2.ports[2])
            b3.move([p2[0] - p1[0], 0])
            R2 = route_basic(port1=b2.ports[2], port2=b3.ports[1], layer=layer)
            r2 = Total.add_ref(R2)

            b4.connect(port=b4.ports[1], destination=b3.ports[2])

            Total.add_port(name=1, port=r1.ports[1])
            Total.add_port(name=2, port=b4.ports[2])
        # third quadrant target
        if (p2[1] < p1[1]) & (p2[0] < p1[0]):
            if bendType == "circular":
                B1 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=0,
                    theta=-90,
                )
                B2 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=-90,
                    theta=-90,
                )
                B3 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=-180,
                    theta=90,
                )
                B4 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=-90,
                    theta=90,
                )
                radiusEff = radius
            if bendType == "gradual":
                B1 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=0,
                    direction="cw",
                )
                B2 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=-90,
                    direction="cw",
                )
                B3 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=-180,
                    direction="ccw",
                )
                B4 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=-90,
                    direction="ccw",
                )
                radiusEff = B1.xsize - width / 2
            b1 = Total.add_ref(B1)
            b2 = Total.add_ref(B2)
            b3 = Total.add_ref(B3)
            b4 = Total.add_ref(B4)

            b1.connect(port=b1.ports[1], destination=Total.ports["t1"])

            b2.connect(port=b2.ports[1], destination=b1.ports[2])
            b2.move([0, p2[1] - p1[1] + radiusEff * 4])
            R1 = route_basic(port1=b1.ports[2], port2=b2.ports[1], layer=layer)
            r1 = Total.add_ref(R1)
            b3.connect(port=b3.ports[1], destination=b2.ports[2])
            b3.move([p2[0] - p1[0], 0])
            R2 = route_basic(port1=b2.ports[2], port2=b3.ports[1], layer=layer)
            r2 = Total.add_ref(R2)

            b4.connect(port=b4.ports[1], destination=b3.ports[2])

            Total.add_port(name=1, port=r1.ports[1])
            Total.add_port(name=2, port=b4.ports[2])
        # fourth quadrant target
        if (p2[1] < p1[1]) & (p2[0] > p1[0]):
            if bendType == "circular":
                B1 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=0,
                    theta=-90,
                )
                B2 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=-90,
                    theta=90,
                )
                radiusEff = radius
            if bendType == "gradual":
                B1 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=0,
                    direction="cw",
                )
                B2 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=-90,
                    direction="ccw",
                )
                radiusEff = B1.xsize - width / 2
            b1 = Total.add_ref(B1)
            b2 = Total.add_ref(B2)

            b1.connect(port=b1.ports[1], destination=Total.ports["t1"])
            b1.move([p2[0] - p1[0] - radiusEff * 2, 0])
            b2.connect(port=b2.ports[1], destination=b1.ports[2])
            b2.move([0, p2[1] - p1[1] + radiusEff * 2])
            R1 = route_basic(port1=Total.ports["t1"], port2=b1.ports[1], layer=layer)
            r1 = Total.add_ref(R1)
            R2 = route_basic(port1=b1.ports[2], port2=b2.ports[1], layer=layer)
            r2 = Total.add_ref(R2)
            Total.add_port(name=1, port=r1.ports[1])
            Total.add_port(name=2, port=b2.ports[2])

    Total.rotate(angle=port1.orientation, center=p1)
    Total.move(origin=Total.ports["t1"], destination=port1)
    return Total


def _route_manhattan90(port1, port2, bendType="circular", layer=0, radius=20):
    # this is a subroutine of route_manhattan() and should not be used by itself.
    Total = Component()
    width = port1.width
    # first map into uniform plane with normal x,y coords
    # allows each situation to be put into uniform cases of quadrants for routing.
    # this is because bends change direction and positioning.
    if port1.orientation == 0:
        p2 = [port2.midpoint[0], port2.midpoint[1]]
        p1 = [port1.midpoint[0], port1.midpoint[1]]
    if port1.orientation == 90:
        p2 = [port2.midpoint[1], -port2.midpoint[0]]
        p1 = [port1.midpoint[1], -port1.midpoint[0]]
    if port1.orientation == 180:
        p2 = [-port2.midpoint[0], -port2.midpoint[1]]
        p1 = [-port1.midpoint[0], -port1.midpoint[1]]
    if port1.orientation == 270:
        p2 = [-port2.midpoint[1], port2.midpoint[0]]
        p1 = [-port1.midpoint[1], port1.midpoint[0]]

    # create placeholder ports based on the imaginary coordinates we created
    Total.add_port(name="t1", midpoint=[0, 0], orientation=0, width=width)

    # CHECK THIS

    # first quadrant target, route upward
    if (p2[1] > p1[1]) & (p2[0] > p1[0]):
        Total.add_port(
            name="t2", midpoint=list(np.subtract(p2, p1)), orientation=-90, width=width
        )
        if bendType == "circular":
            B1 = _arc(
                radius=radius,
                width=width,
                layer=layer,
                angle_resolution=1,
                start_angle=0,
                theta=90,
            )
            radiusEff = radius
        if bendType == "gradual":
            B1 = _gradual_bend(
                radius=radius, width=width, layer=layer, start_angle=0, direction="ccw"
            )
            radiusEff = B1.xsize - width / 2
        b1 = Total.add_ref(B1)
        b1.connect(port=b1.ports[1], destination=Total.ports["t1"])
        b1.move([p2[0] - p1[0] - radiusEff, 0])

        R1 = route_basic(port1=Total.ports["t1"], port2=b1.ports[1], layer=layer)
        R2 = route_basic(port1=b1.ports[2], port2=Total.ports["t2"], layer=layer)
        r1 = Total.add_ref(R1)
        r2 = Total.add_ref(R2)
        Total.add_port(name=1, port=r1.ports[1])
        Total.add_port(name=2, port=r2.ports[2])

    # fourth quadrant target, route downward
    if (p2[1] < p1[1]) & (p2[0] > p1[0]):
        Total.add_port(
            name="t2", midpoint=list(np.subtract(p2, p1)), orientation=90, width=width
        )
        if bendType == "circular":
            B1 = _arc(
                radius=radius,
                width=width,
                layer=layer,
                angle_resolution=1,
                start_angle=0,
                theta=-90,
            )
            radiusEff = radius
        if bendType == "gradual":
            B1 = _gradual_bend(
                radius=radius, width=width, layer=layer, start_angle=0, direction="cw"
            )
            radiusEff = B1.xsize - width / 2
        b1 = Total.add_ref(B1)
        b1.connect(port=b1.ports[1], destination=Total.ports["t1"])
        b1.move([p2[0] - p1[0] - radiusEff, 0])
        R1 = route_basic(port1=Total.ports["t1"], port2=b1.ports[1], layer=layer)
        R2 = route_basic(port1=b1.ports[2], port2=Total.ports["t2"], layer=layer)
        r1 = Total.add_ref(R1)
        r2 = Total.add_ref(R2)
        Total.add_port(name=1, port=r1.ports[1])
        Total.add_port(name=2, port=r2.ports[2])
    Total.rotate(angle=port1.orientation, center=p1)
    Total.move(origin=Total.ports["t1"], destination=port1)

    return Total


def route_manhattan(
    port1: Port,
    port2: Port,
    bendType: str = "gradual",
    layer: Optional[Layer] = None,
    radius: float = TECH.waveguide.strip.radius,
):
    """Returns Route along cardinal directions between two ports
    placed diagonally from each other

    Args:
        port1:
        port2:
        bendType: gradual, circular

    """
    layer = layer or port1.layer

    valid_bend_types = ["circular", "gradual"]

    if bendType not in valid_bend_types:
        raise ValueError(f"bendType={bendType} not in {valid_bend_types}")

    if bendType == "gradual":
        b = _gradual_bend(radius=radius)
        radius_eff = b.xsize
    else:
        radius_eff = radius

    if (
        abs(port1.midpoint[0] - port2.midpoint[0]) < 2 * radius_eff
        or abs(port1.midpoint[1] - port2.midpoint[1]) < 2 * radius_eff
    ):
        raise RoutingError(
            f"bend does not fit (radius = {radius_eff}) you need radius <",
            min(
                [
                    abs(port1.midpoint[0] - port2.midpoint[0]) / 2,
                    abs(port1.midpoint[1] - port2.midpoint[1]) / 2,
                ]
            ),
        )

    Total = Component()
    references = []

    width = port1.width
    # first map into uniform plane with normal x,y coords
    # allows each situation to be put into uniform cases of quadrants for routing.
    # this is because bends change direction and positioning.
    if port1.orientation == 0:
        p2 = [port2.midpoint[0], port2.midpoint[1]]
        p1 = [port1.midpoint[0], port1.midpoint[1]]
    if port1.orientation == 90:
        p2 = [port2.midpoint[1], -port2.midpoint[0]]
        p1 = [port1.midpoint[1], -port1.midpoint[0]]
    if port1.orientation == 180:
        p2 = [-port2.midpoint[0], -port2.midpoint[1]]
        p1 = [-port1.midpoint[0], -port1.midpoint[1]]
    if port1.orientation == 270:
        p2 = [-port2.midpoint[1], port2.midpoint[0]]
        p1 = [-port1.midpoint[1], port1.midpoint[0]]

    Total.add_port(name=1, port=port1)
    Total.add_port(name=2, port=port2)

    ports = {1: Total.ports[1], 2: Total.ports[2]}

    if p2[1] == p1[1] or p2[0] == p1[0]:
        raise RoutingError("Error - ports must be at different x AND y values.")

    # if it is parallel or anti-parallel, route with 180 option
    if (
        np.round(np.abs(np.mod(port1.orientation - port2.orientation, 360)), 3) == 180
    ) or (np.round(np.abs(np.mod(port1.orientation - port2.orientation, 360)), 3) == 0):
        R1 = _route_manhattan180(
            port1=port1, port2=port2, bendType=bendType, layer=layer, radius=radius
        )
        r1 = Total.add_ref(R1)
        references.append(r1)

    else:
        # first quadrant case
        if (p2[1] > p1[1]) & (p2[0] > p1[0]):
            # simple 90 degree single-bend case
            if (
                port2.orientation == port1.orientation - 90
                or port2.orientation == port1.orientation + 270
            ):
                R1 = _route_manhattan90(
                    port1=port1,
                    port2=port2,
                    bendType=bendType,
                    layer=layer,
                    radius=radius,
                )
                r1 = Total.add_ref(R1)
                references.append(r1)
            elif (
                port2.orientation == port1.orientation + 90
                or port2.orientation == port1.orientation - 270
            ):
                if bendType == "circular":
                    B1 = _arc(
                        radius=radius,
                        width=width,
                        layer=layer,
                        angle_resolution=1,
                        start_angle=port1.orientation,
                        theta=90,
                    )
                if bendType == "gradual":
                    B1 = _gradual_bend(
                        radius=radius,
                        width=width,
                        layer=layer,
                        start_angle=port1.orientation,
                        direction="ccw",
                    )
                b1 = Total.add_ref(B1)
                references.append(b1)
                b1.connect(port=1, destination=port1)

                R1 = _route_manhattan180(
                    port1=b1.ports[2],
                    port2=port2,
                    bendType=bendType,
                    layer=layer,
                    radius=radius,
                )
                r1 = Total.add_ref(R1)
                references.append(r1)
        # second quadrant case
        if (p2[1] > p1[1]) & (p2[0] < p1[0]):
            if (
                np.abs(port1.orientation - port2.orientation) == 90
                or np.abs(port1.orientation - port2.orientation) == 270
            ):
                if bendType == "circular":
                    B1 = _arc(
                        radius=radius,
                        width=width,
                        layer=layer,
                        angle_resolution=1,
                        start_angle=port1.orientation,
                        theta=90,
                    )
                if bendType == "gradual":
                    B1 = _gradual_bend(
                        radius=radius,
                        width=width,
                        layer=layer,
                        start_angle=port1.orientation,
                        direction="ccw",
                    )
                b1 = Total.add_ref(B1)
                b1.connect(port=1, destination=port1)
                references.append(b1)
                R1 = _route_manhattan180(
                    port1=b1.ports[2],
                    port2=port2,
                    bendType=bendType,
                    layer=layer,
                    radius=radius,
                )
                r1 = Total.add_ref(R1)
                references.append(r1)
        # third quadrant case
        if (p2[1] < p1[1]) & (p2[0] < p1[0]):
            if (
                np.abs(port1.orientation - port2.orientation) == 90
                or np.abs(port1.orientation - port2.orientation) == 270
            ):
                if bendType == "circular":
                    B1 = _arc(
                        radius=radius,
                        width=width,
                        layer=layer,
                        angle_resolution=1,
                        start_angle=port1.orientation,
                        theta=-90,
                    )
                    # radiusEff = radius
                if bendType == "gradual":
                    B1 = _gradual_bend(
                        radius=radius,
                        width=width,
                        layer=layer,
                        start_angle=port1.orientation,
                        direction="cw",
                    )
                    # radiusEff = B1.xsize - width / 2
                b1 = Total.add_ref(B1)
                b1.connect(port=1, destination=port1)
                references.append(b1)
                R1 = _route_manhattan180(
                    port1=b1.ports[2],
                    port2=port2,
                    bendType=bendType,
                    layer=layer,
                    radius=radius,
                )
                r1 = Total.add_ref(R1)
                references.append(r1)
        # fourth quadrant case
        if (p2[1] < p1[1]) & (p2[0] > p1[0]):
            # simple 90 degree single-bend case
            if (
                port2.orientation == port1.orientation + 90
                or port2.orientation == port1.orientation - 270
            ):
                R1 = _route_manhattan90(
                    port1=port1,
                    port2=port2,
                    bendType=bendType,
                    layer=layer,
                    radius=radius,
                )
                r1 = Total.add_ref(R1)
                references.append(r1)
            elif (
                port2.orientation == port1.orientation - 90
                or port2.orientation == port1.orientation + 270
            ):
                if bendType == "circular":
                    B1 = _arc(
                        radius=radius,
                        width=width,
                        layer=layer,
                        angle_resolution=1,
                        start_angle=port1.orientation,
                        theta=-90,
                    )
                    # radiusEff = radius
                if bendType == "gradual":
                    B1 = _gradual_bend(
                        radius=radius,
                        width=width,
                        layer=layer,
                        start_angle=port1.orientation,
                        direction="cw",
                    )
                    # radiusEff = B1.xsize - width / 2
                b1 = Total.add_ref(B1)
                b1.connect(port=1, destination=port1)
                references.append(b1)
                R1 = _route_manhattan180(
                    port1=b1.ports[2],
                    port2=port2,
                    bendType=bendType,
                    layer=layer,
                    radius=radius,
                )
                r1 = Total.add_ref(R1)
                references.append(r1)

    references = []
    length = 0
    for ref1 in Total.references:
        for ref2 in ref1.parent.references:
            references.append(ref2)
            length += ref2.info["length"]

    ports = (Total.ports[1], Total.ports[2])
    length = snap_to_grid(length)
    return Route(references=references, ports=ports, length=length)


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component("test_route_manhattan_circular")
    pitch = 9.0
    ys1 = [0, 10, 20]
    N = len(ys1)
    ys2 = [15 + i * pitch for i in range(N)]

    ports1 = [gf.Port(f"L_{i}", (0, ys1[i]), 0.5, 0) for i in range(N)]
    ports2 = [gf.Port(f"R_{i}", (20, ys2[i]), 0.5, 180) for i in range(N)]

    ports1 = [
        # gf.Port("in1", (10, 5), 0.5, 90),
        # gf.Port("in2", (-10, 20), 0.5, 0),
        # gf.Port("in3", (10, 30), 0.5, 0),
        # gf.Port("in4", (-10, -5), 0.5, 90),
        gf.Port("in5", (0, 0), 0.5, 0),
        # gf.Port("in6", (0, 0), 0.5, 0),
    ]

    ports2 = [
        # gf.Port("in1", (90, -60), 0.5, 180),
        # gf.Port("in2", (-100, 20), 0.5, 0),
        # gf.Port("in3", (100, -25), 0.5, 0),
        # gf.Port("in4", (-150, -65), 0.5, 270),
        gf.Port("in5", (15, 6), 0.5, 180),
        # gf.Port("in6", (0, 12), 0.5, 180),
    ]
    N = len(ports1)

    for i in range(N):
        # route = route_manhattan(ports1[i], ports2[i], radius=3, bendType="circular")
        route = route_manhattan(ports1[i], ports2[i], radius=1, bendType="gradual")
        c.add(route.references)
    # references = route_basic(port1=ports1[i], port2=ports2[i])
    # print(route.length)

    # c = _gradual_bend()
    # c = _arc(theta=20)
    c.show(show_ports=True)
