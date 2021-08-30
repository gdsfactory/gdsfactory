from typing import Optional

import numpy as np
import scipy.optimize as so
from numpy import float64

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bezier import (
    bezier,
    bezier_curve,
    find_min_curv_bezier_control_points,
)
from gdsfactory.components.ellipse import ellipse
from gdsfactory.components.taper import taper
from gdsfactory.geo_utils import path_length
from gdsfactory.tech import LAYER
from gdsfactory.types import ComponentFactory, ComponentOrFactory, Layer


def snap_to_grid(p: float, grid_per_unit: int = 1000) -> float64:
    """round"""
    return np.round(p * grid_per_unit) / grid_per_unit


@cell
def crossing_arm(
    width: float = 0.5,
    r1: float = 3.0,
    r2: float = 1.1,
    w: float = 1.2,
    L: float = 3.4,
) -> Component:
    """arm of a crossing"""
    c = Component()
    _ellipse = ellipse(radii=(r1, r2), layer=LAYER.SLAB150).ref()
    c.add(_ellipse)
    c.absorb(_ellipse)

    a = np.round(L + w / 2, 3)
    h = width / 2

    taper_pts = [
        (-a, h),
        (-w / 2, w / 2),
        (w / 2, w / 2),
        (a, h),
        (a, -h),
        (w / 2, -w / 2),
        (-w / 2, -w / 2),
        (-a, -h),
    ]

    c.add_polygon(taper_pts, layer=LAYER.WG)
    c.add_port(
        name="o1", midpoint=(-a, 0), orientation=180, width=width, layer=LAYER.WG
    )

    c.add_port(name="o2", midpoint=(a, 0), orientation=0, width=width, layer=LAYER.WG)

    return c


@cell
def crossing(arm: ComponentFactory = crossing_arm) -> Component:
    """Waveguide crossing"""
    cx = Component()
    arm = arm() if callable(arm) else arm
    arm_h = arm.ref()
    arm_v = arm.ref(rotation=90)

    port_id = 0
    for c in [arm_h, arm_v]:
        cx.add(c)
        cx.absorb(c)
        for p in c.ports.values():
            cx.add_port(name=port_id, port=p)
            port_id += 1
    cx.auto_rename_ports()
    return cx


@cell
def crossing_from_taper(taper=lambda: taper(width2=2.5, length=3.0)):
    """
    Crossing based on a taper. The default is a dummy taper
    """
    taper = taper() if callable(taper) else taper

    c = Component()
    for i, a in enumerate([0, 90, 180, 270]):
        _taper = taper.ref(position=(0, 0), port_id="o2", rotation=a)
        c.add(_taper)
        c.add_port(name=i, port=_taper.ports["o1"])
        c.absorb(_taper)

    c.auto_rename_ports()
    return c


@cell
def crossing_etched(
    width: float = 0.5,
    r1: float = 3.0,
    r2: float = 1.1,
    w: float = 1.2,
    L: float = 3.4,
    layer_wg: Layer = LAYER.WG,
    layer_slab: Layer = LAYER.SLAB150,
):
    """
    Waveguide crossing:
    - The full crossing has to be on WG layer (to start with a 220nm slab)
    - Then we etch the ellipses down to 150nm slabs and we keep linear taper at 220nm.
    What we write is what we etch on this step
    """

    # Draw the ellipses
    c = Component()
    _ellipse1 = c << ellipse(radii=(r1, r2), layer=layer_wg)
    _ellipse2 = c << ellipse(radii=(r2, r1), layer=layer_wg)
    c.absorb(_ellipse1)
    c.absorb(_ellipse2)

    a = L + w / 2
    h = width / 2

    taper_cross_pts = [
        (-a, h),
        (-w / 2, w / 2),
        (-h, a),
        (h, a),
        (w / 2, w / 2),
        (a, h),
        (a, -h),
        (w / 2, -w / 2),
        (h, -a),
        (-h, -a),
        (-w / 2, -w / 2),
        (-a, -h),
    ]

    c.add_polygon(taper_cross_pts, layer=layer_wg)

    # tapers_poly = c.add_polygon(taper_cross_pts, layer=layer_wg)
    # b = a - 0.1  # To make sure we get 4 distinct polygons when doing bool ops
    # tmp_polygon = [(-b, b), (b, b), (b, -b), (-b, -b)]
    # polys_etch = gdspy.fast_boolean([tmp_polygon], tapers_poly, "not", layer=layer_slab)
    # c.add(polys_etch)

    positions = [(a, 0), (0, a), (-a, 0), (0, -a)]
    angles = [0, 90, 180, 270]

    i = 0
    for p, angle in zip(positions, angles):
        c.add_port(
            name=str(i),
            midpoint=p,
            orientation=angle,
            width=width,
            layer=layer_wg,
        )
        i += 1

    c.auto_rename_ports()
    return c


@cell
def crossing45(
    crossing: ComponentFactory = crossing,
    port_spacing: float = 40.0,
    dx: Optional[float] = None,
    alpha: float = 0.08,
    npoints: int = 101,
) -> Component:
    r"""Returns 45deg crossing with bends.

    Args:
        crossing: 90D crossing
        port_spacing: target I/O port spacing
        dx: target length
        alpha: optimization parameter. Try with 0.1 to start with.
            - If the structure has too tight bends, diminish it.
            - If raise assertion angle errors, increase it
        npoints: number of points.


    Implementation note: The 45 Degree crossing CANNOT be kept as an SRef since
    we only allow for multiples of 90Deg rotations in SRef

    .. code::

        ----   ----
            \ /
             X
            / \
        ---    ----

    """

    crossing = crossing() if callable(crossing) else crossing

    c = Component()
    _crossing = crossing.ref(rotation=45)
    c.add(_crossing)

    # Add bends
    p_e = _crossing.ports["o3"].midpoint
    p_w = _crossing.ports["o1"].midpoint
    p_n = _crossing.ports["o2"].midpoint
    p_s = _crossing.ports["o4"].midpoint

    # Flatten the crossing - not an SRef anymore
    c.absorb(_crossing)
    dx = dx or port_spacing
    dy = port_spacing / 2

    start_angle = 45
    end_angle = 0
    cpts = find_min_curv_bezier_control_points(
        start_point=p_e,
        end_point=(dx, dy),
        start_angle=start_angle,
        end_angle=end_angle,
        npoints=npoints,
        alpha=alpha,
    )

    bend = bezier(
        control_points=cpts,
        start_angle=start_angle,
        end_angle=end_angle,
        npoints=npoints,
    )

    tol = 1e-2
    assert abs(bend.info["start_angle"] - start_angle) < tol, bend.info["start_angle"]
    assert abs(bend.info["end_angle"] - end_angle) < tol, bend.info["end_angle"]

    b_tr = bend.ref(position=p_e, port_id="o1")
    b_tl = bend.ref(position=p_n, port_id="o1", h_mirror=True)
    b_bl = bend.ref(position=p_w, port_id="o1", rotation=180)
    b_br = bend.ref(position=p_s, port_id="o1", v_mirror=True)

    for cmp_ref in [b_tr, b_br, b_tl, b_bl]:
        # cmp_ref = _cmp.ref()
        c.add(cmp_ref)
        c.absorb(cmp_ref)

    c.info["components"] = {"bezier_bend": bend, "crossing": crossing}
    c.info["min_bend_radius"] = b_br.info["min_bend_radius"]

    c.add_port("o1", port=b_br.ports["o2"])
    c.add_port("o2", port=b_tr.ports["o2"])
    c.add_port("o3", port=b_bl.ports["o2"])
    c.add_port("o4", port=b_tl.ports["o2"])
    c.snap_ports_to_grid()
    c.auto_rename_ports()
    return c


@cell
def compensation_path(
    crossing45: ComponentOrFactory = crossing45, direction: str = "top"
) -> Component:
    r"""Returns Component Path with same path length as the crossing

    with input and output ports having same y coordinates

    Args:
        crossing45: the crossing45 component that we want to match in path length
            This component needs to have .info["components"] with bends and crossing
        direction: the direction in which the bend should go "top" / "bottom"


    .. code::

          ----       ----
              \     /
               \   /
                \ /
                 X
                / \
               /   \
              /     \
          ----       ----

    Compensation path:

    .. code::

             --+--
           _/     \_
        --/         \--


    """
    # Get total path length taken by the bends
    crossing45 = crossing45() if callable(crossing45) else crossing45
    X45_cmps = crossing45.info["components"]
    length = 2 * X45_cmps["bezier_bend"].info["length"]

    # Find a bezier S-bend with half this length, but with a fixed length
    # governed by the crossing45 X-distance (west to east ports) and
    # the crossing x_distance

    target_bend_length = length / 2

    def get_x_span(cmp):
        return cmp.ports["o3"].x - cmp.ports["o1"].x

    x_span_crossing45 = get_x_span(crossing45)
    x_span_crossing = get_x_span(X45_cmps["crossing"])

    # x span allowed for the bend
    x0 = (x_span_crossing45 - x_span_crossing) / 2

    def get_control_pts(x, y):
        return [(0, 0), (x0 / 2, 0), (x0 / 2, y), (x0, y)]

    def f(y):
        control_points = get_control_pts(x0, y)
        t = np.linspace(0, 1, 51)
        path_points = bezier_curve(t, control_points)
        return path_length(path_points) - target_bend_length

    # the path length of the s-bend between two ports p0 and p1 is :
    # - larger than the euclidian distance L2(p0, p1)
    # - smaller than the manhattan distance DL(p0, p1)
    #
    # This gives the bounds for the brentq root finding

    ya = target_bend_length - x0
    yb = np.sqrt(target_bend_length ** 2 - x0 ** 2)

    solution = so.root_scalar(f, bracket=[ya, yb], method="brentq")

    y_bend = solution.root
    y_bend = snap_to_grid(y_bend)

    if direction == "top":
        v_mirror = False
    else:
        v_mirror = True

    sbend = bezier(control_points=get_control_pts(x0, y_bend))

    component = Component()
    crossing0 = X45_cmps["crossing"].ref()
    component.add(crossing0)

    sbend_left = sbend.ref(
        position=crossing0.ports["o1"], port_id="o2", v_mirror=v_mirror
    )
    sbend_right = sbend.ref(
        position=crossing0.ports["o3"], port_id="o2", h_mirror=True, v_mirror=v_mirror
    )

    component.add(sbend_left)
    component.add(sbend_right)

    component.add_port("o1", port=sbend_left.ports["o1"])
    component.add_port("o2", port=sbend_right.ports["o1"])

    component.info["min_bend_radius"] = sbend.info["min_bend_radius"]
    component.info["components"] = {"sbend": sbend}
    return component


def _demo():
    """plot curvature of bends"""
    from matplotlib import pyplot as plt

    c = crossing45(port_spacing=20.0, dx=15)
    c2 = compensation_path(crossing45=c)
    print(c.info["min_bend_radius"])
    print(c2.info["min_bend_radius"])

    component = Component(name="top_lvl")
    component.add(c.ref(port_id="o1"))
    component.add(c2.ref(port_id="o1", position=(0, 10)))

    bend_info1 = c.info["components"]["bezier_bend"].info
    bend_info2 = c2.info["components"]["sbend"].info

    DL = bend_info1["length"]
    L2 = bend_info1["length"]
    plt.plot(bend_info1["t"][1:-1] * DL, abs(bend_info1["curvature"]))
    plt.plot(bend_info2["t"][1:-1] * L2, abs(bend_info2["curvature"]))
    plt.xlabel("bend length (um)")
    plt.ylabel("curvature (um^-1)")
    component.show()
    plt.show()


if __name__ == "__main__":
    c = compensation_path()
    # c = crossing()
    # c = crossing45(port_spacing=40)
    # print(c.ports["E1"].y - c.ports['o2'].y)
    # print(c.get_ports_array())
    # _demo()
    # c = crossing_from_taper()
    # c.pprint
    # c = crossing_etched()
    c.show()
