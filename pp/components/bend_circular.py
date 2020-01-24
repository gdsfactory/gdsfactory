from numpy import pi, cos, sin
import numpy as np

import pp
from pp.layers import LAYER

__version__ = "0.0.1"


def _interpolate_segment(p0, p1, N=2):
    p0 = np.array(p0)
    p1 = np.array(p1)
    dp = p1 - p0
    d = np.sqrt(dp[0] ** 2 + dp[1] ** 2)
    return [p0 + dp * a / d for a in np.linspace(0, d, N)]


def _bend_path_from_pts(pts, n_interp=2):
    N = len(pts)
    pts0 = pts[: N // 2]
    pts1 = pts[N // 2 :][::-1]

    pts = [((x0 + x1) * 0.5, (y0 + y1) * 0.5) for (x0, y0), (x1, y1) in zip(pts0, pts1)]

    all_pts = [pts[0]]
    for p0, p1 in zip(pts[:-1], pts[1:]):
        _pts = _interpolate_segment(p0, p1, N=n_interp)
        all_pts += _pts[1:]

    return all_pts


def _bend_path(radius=10.0, start_angle=0, theta=-90, angle_resolution=2.5):
    angle1 = (start_angle) * pi / 180
    angle2 = (start_angle + theta) * pi / 180
    t = np.linspace(angle1, angle2, int(abs(theta) / angle_resolution))
    points_x = (radius * cos(t)).tolist()
    points_y = (radius * sin(t)).tolist()
    return points_x, points_y


def _bend_points(
    radius=10.0,
    width=0.5,
    theta=-90,
    start_angle=0,
    angle_resolution=2.5,
    inner_radius=None,
    outer_radius=None,
):
    if inner_radius == None:
        inner_radius = radius - width / 2
    if outer_radius == None:
        outer_radius = radius + width / 2
    angle1 = (start_angle) * pi / 180
    angle2 = (start_angle + theta) * pi / 180
    t = np.linspace(angle1, angle2, int(abs(theta) / angle_resolution))
    inner_points_x = (inner_radius * cos(t)).tolist()
    inner_points_y = (inner_radius * sin(t)).tolist()
    outer_points_x = (outer_radius * cos(t)).tolist()
    outer_points_y = (outer_radius * sin(t)).tolist()
    xpts = inner_points_x + outer_points_x[::-1]
    ypts = inner_points_y + outer_points_y[::-1]
    return xpts, ypts


def _disk_section_points(
    radius=10.0, theta=-90, start_angle=0, angle_resolution=2.5, layer=LAYER.WG
):
    angle1 = (start_angle) * pi / 180
    angle2 = (start_angle + theta) * pi / 180
    t = np.linspace(angle1, angle2, int(abs(theta) / angle_resolution))
    xpts = (radius * cos(t)).tolist()
    ypts = (radius * sin(t)).tolist()
    xpts.append(0)
    ypts.append(0)
    return xpts, ypts


@pp.autoname
def bend_circular(
    radius=10.0,
    width=0.5,
    theta=-90,
    start_angle=0,
    angle_resolution=2.5,
    layer=LAYER.WG,
    cladding_layer=pp.layer("wgclad"),
    cladding_offset=3,
):
    """ Creates an arc of arclength ``theta`` starting at angle ``start_angle``

    Args:
        radius
        width: of the waveguide
        theta: arc length
        start_angle:
        angle_resolution
        layer

    .. plot::
      :include-source:

      import pp

      c = pp.c.bend_circular(
        radius=10,
        width=0.5,
        theta=-90,
        start_angle=0,
      )
      pp.plotgds(c)
    """
    component = pp.Component()

    # Core
    inner_radius = radius - width / 2
    outer_radius = radius + width / 2
    angle1 = (start_angle) * pi / 180
    angle2 = (start_angle + theta) * pi / 180
    t = np.linspace(angle1, angle2, int(abs(theta) / angle_resolution))
    inner_points_x = (inner_radius * cos(t)).tolist()
    inner_points_y = (inner_radius * sin(t)).tolist()
    outer_points_x = (outer_radius * cos(t)).tolist()
    outer_points_y = (outer_radius * sin(t)).tolist()
    xpts = inner_points_x + outer_points_x[::-1]
    ypts = inner_points_y + outer_points_y[::-1]

    component.add_polygon(points=(xpts, ypts), layer=layer)

    # Cladding
    w = width + 2 * cladding_offset
    inner_radius = radius - w / 2
    outer_radius = radius + w / 2
    angle1 = (start_angle) * pi / 180
    angle2 = (start_angle + theta) * pi / 180
    t = np.linspace(angle1, angle2, int(abs(theta) / angle_resolution))
    inner_points_x = (inner_radius * cos(t)).tolist()
    inner_points_y = (inner_radius * sin(t)).tolist()
    outer_points_x = (outer_radius * cos(t)).tolist()
    outer_points_y = (outer_radius * sin(t)).tolist()
    xpts = inner_points_x + outer_points_x[::-1]
    ypts = inner_points_y + outer_points_y[::-1]

    if cladding_layer is not None:
        component.add_polygon(points=(xpts, ypts), layer=cladding_layer)

    component.add_port(
        name="W0",
        midpoint=(radius * cos(angle1), radius * sin(angle1)),
        width=width,
        orientation=start_angle - 90 + 180 * (theta < 0),
        layer=layer,
    )
    component.add_port(
        name="N0",
        midpoint=(radius * cos(angle2), radius * sin(angle2)),
        width=width,
        orientation=start_angle + theta + 90 - 180 * (theta < 0),
        layer=layer,
    )
    component.info["length"] = (abs(theta) * pi / 180) * radius
    component.radius = radius
    component.width = width
    component.move((0, radius))

    pp.ports.port_naming.rename_ports_by_orientation(component)
    return component


@pp.autoname
def bend_circular_deep_rib(
    layer=pp.layer("slab90"), cladding_layer=pp.layer("slab90clad"), **kwargs
):
    c = bend_circular(layer=layer, cladding_layer=cladding_layer, **kwargs)
    pp.ports.port_naming.rename_ports_by_orientation(c)
    return c


@pp.autoname
def bend_circular_shallow_rib(
    layer=pp.layer("slab150"), cladding_layer=pp.layer("slab150clad"), **kwargs
):
    return bend_circular(layer=layer, cladding_layer=cladding_layer, **kwargs)


@pp.autoname
def _bend_circular(
    radius=10.0,
    width=0.5,
    theta=-90,
    start_angle=0,
    angle_resolution=2.5,
    layer=LAYER.WG,
):

    component = pp.Component()

    xpts, ypts = _bend_points(radius, width, theta, start_angle, angle_resolution)
    angle1 = (start_angle) * pi / 180
    angle2 = (start_angle + theta) * pi / 180

    component.add_polygon(points=(xpts, ypts), layer=layer)

    component.add_port(
        name="W0",
        midpoint=(radius * cos(angle1), radius * sin(angle1)),
        width=width,
        orientation=start_angle - 90 + 180 * (theta < 0),
        layer=layer,
    )
    component.add_port(
        name="N0",
        midpoint=(radius * cos(angle2), radius * sin(angle2)),
        width=width,
        orientation=start_angle + theta + 90 - 180 * (theta < 0),
        layer=layer,
    )
    component.info["length"] = (abs(theta) * pi / 180) * radius
    component.radius = radius
    component.width = width
    component.move((0, radius))

    pp.ports.auto_rename_ports(component)
    return component


@pp.autoname
def bend_circular180(
    radius=10.0,
    width=0.5,
    theta=180,
    start_angle=-90,
    angle_resolution=2.5,
    layer=LAYER.WG,
    **kwargs
):
    c = bend_circular(
        radius=radius,
        width=width,
        theta=theta,
        start_angle=start_angle,
        angle_resolution=angle_resolution,
        layer=layer,
        **kwargs
    )
    return c


def _bend_circular_windows(
    radius=10,
    start_angle=0,
    theta=-90,
    angle_resolution=2.5,
    windows=[-0.25, 0.25, LAYER.WG],
):
    """
    windows: [(y_start, y_stop, layer), ...]
    """
    component = pp.Component()
    y_min, y_max, layer0 = windows[0]
    y_min, y_max = min(y_min, y_max), max(y_min, y_max)

    bend_params = {
        "start_angle": start_angle,
        "theta": theta,
        "angle_resolution": angle_resolution,
    }

    # Create each bend shape
    for y_start, y_stop, layer in windows:
        w = abs(y_stop - y_start)
        y = (y_stop + y_start) / 2
        r = radius - y
        _c = bend_circular(radius=r, width=w, layer=layer, **bend_params).ref()
        _c.movey(y)
        component.add(_c)
        component.absorb(_c)
        y_min = min(y_stop, y_start, y_min)
        y_max = max(y_stop, y_start, y_max)
    width = y_max - y_min

    # Define ports
    angle1 = (start_angle) * pi / 180
    angle2 = (start_angle + theta) * pi / 180

    component.add_port(
        name="tmp0",
        midpoint=(radius * cos(angle1), radius + radius * sin(angle1)),
        width=width,
        orientation=start_angle - 90 + 180 * (theta < 0),
        layer=layer0,
    )
    component.add_port(
        name="tmp1",
        midpoint=(radius * cos(angle2), radius + radius * sin(angle2)),
        width=width,
        orientation=start_angle + theta + 90 - 180 * (theta < 0),
        layer=layer0,
    )
    pp.ports.auto_rename_ports(component)
    return component


@pp.autoname
def bend_circular_trenches(
    width=0.5, trench_width=3.0, trench_offset=0.2, trench_layer=LAYER.SLAB90, **kwargs
):
    """ defines trenches """
    w = width / 2
    ww = w + trench_width
    wt = ww + trench_offset
    windows = [(-ww, ww, LAYER.WG), (-wt, -w, trench_layer), (w, wt, trench_layer)]
    return _bend_circular_windows(windows=windows, **kwargs)


@pp.autoname
def bend_circular_slab(width=0.5, cladding=2.0, slab_layer=LAYER.SLAB90, **kwargs):
    a = width / 2
    b = a + cladding
    windows = [(-a, a, LAYER.WG), (-b, b, slab_layer)]
    return _bend_circular_windows(windows=windows, **kwargs)


bend_circular_ridge = bend_circular_trenches


@pp.autoname
def bend_circular_slot(width=0.5, gap=0.2, **kwargs):
    a = width / 2
    d = a + gap / 2
    windows = [(-a - d, a - d, LAYER.WG), (-a + d, a + d, LAYER.WG)]
    return _bend_circular_windows(windows=windows, **kwargs)


def _demo_bend():
    c = bend_circular()
    pp.write_gds(c)


if __name__ == "__main__":
    # c = bend_circular_trenches()
    c = bend_circular_deep_rib()
    print(c.ports)
    # c = bend_circular180()
    # c = bend_circular_slot()
    # c = bend_circular(start_angle=180, theta=-180)
    pp.show(c)
