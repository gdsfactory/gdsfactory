"""Waveguide crossings."""

from __future__ import annotations

from functools import partial

import numpy as np
from numpy import float64

import gdsfactory as gf
from gdsfactory import cell
from gdsfactory.component import Component
from gdsfactory.components.bezier import (
    bezier,
    find_min_curv_bezier_control_points,
)
from gdsfactory.components.ellipse import ellipse
from gdsfactory.components.taper import taper
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec


def snap_to_grid(p: float, grid_per_unit: int = 1000) -> float64:
    """Round."""
    return np.round(p * grid_per_unit) / grid_per_unit


@cell
def crossing_arm(
    r1: float = 3.0,
    r2: float = 1.1,
    w: float = 1.2,
    L: float = 3.4,
    layer_slab: LayerSpec = "SLAB150",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns crossing arm.

    Args:
        r1: ellipse radius1.
        r2: ellipse radius2.
        w: width in um.
        L: length in um.
        layer_slab: for the shallow etch.
        cross_section: spec.
    """
    c = Component()

    layer_slab = gf.get_layer(layer_slab)
    c << ellipse(radii=(r1, r2), layer=layer_slab)

    xs = gf.get_cross_section(cross_section)
    width = xs.width
    layer_wg = gf.get_layer(xs.layer)

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

    c.add_polygon(taper_pts, layer=layer_wg)
    c.add_port(
        name="o1",
        center=(-a, 0),
        orientation=180,
        width=width,
        layer=layer_wg,
        cross_section=xs,
    )

    c.add_port(
        name="o2",
        center=(a, 0),
        orientation=0,
        width=width,
        layer=layer_wg,
        cross_section=xs,
    )

    return c


@cell
def crossing(
    arm: ComponentSpec = crossing_arm,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Waveguide crossing.

    Args:
        arm: arm spec.
        cross_section: spec.
    """
    x = gf.get_cross_section(cross_section)
    c = Component()
    arm = gf.get_component(arm)
    port_id = 0
    for rotation in [0, 90]:
        ref = c << arm
        ref.drotate(rotation)
        for p in ref.ports:
            c.add_port(name=port_id, port=p)
            port_id += 1

    c.auto_rename_ports()
    x.add_bbox(c)
    c.flatten()
    return c


_taper = partial(taper, width2=2.5, length=3)


@cell
def crossing_from_taper(taper=_taper) -> Component:
    """Returns Crossing based on a taper.

    The default is a dummy taper.

    Args:
        taper: taper function.
    """
    taper = gf.get_component(taper)

    c = Component()
    for i, a in enumerate([0, 90, 180, 270]):
        # _taper = taper.ref(position=(0, 0), port_id="o2", rotation=a)
        # c.add(_taper)
        _taper = c << taper
        _taper.drotate(a, center=gf.kdb.DPoint(*_taper["o2"].dcenter))
        c.add_port(name=i, port=_taper.ports["o1"])

    c.auto_rename_ports()
    c.flatten()
    return c


@cell
def crossing_etched(
    width: float = 0.5,
    r1: float = 3.0,
    r2: float = 1.1,
    w: float = 1.2,
    L: float = 3.4,
    layer_wg: LayerSpec = "WG",
    layer_slab: LayerSpec = "SLAB150",
) -> Component:
    """Waveguide crossing.

    Full crossing has to be on WG layer (to start with a 220nm slab).
    Then we etch the ellipses down to 150nm slabs and we keep linear taper at 220nm.

    Args:
        width: input waveguides width.
        r1: radii.
        r2: radii.
        w: wide width.
        L: length.
        layer_wg: waveguide layer.
        layer_slab: shallow etch layer.
    """
    layer_wg = gf.get_layer(layer_wg)
    layer_slab = gf.get_layer(layer_slab)

    # Draw the ellipses
    c = Component()
    _ = c << ellipse(radii=(r1, r2), layer=layer_wg)
    _ = c << ellipse(radii=(r2, r1), layer=layer_wg)

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
    # polys_etch = gdstk.fast_boolean([tmp_polygon], tapers_poly, "not", layer=layer_slab)
    # c.add(polys_etch)

    positions = [(a, 0), (0, a), (-a, 0), (0, -a)]
    angles = [0, 90, 180, 270]

    for i, (p, angle) in enumerate(zip(positions, angles)):
        c.add_port(
            name=str(i),
            center=p,
            orientation=angle,
            width=width,
            layer=layer_wg,
        )
    c.auto_rename_ports()
    c.flatten()
    return c


@cell
def crossing45(
    crossing: ComponentSpec = crossing,
    port_spacing: float = 40.0,
    dx: float | None = None,
    alpha: float = 0.08,
    npoints: int = 101,
    cross_section: CrossSectionSpec = "strip",
    cross_section_bends: CrossSectionSpec = "strip",
) -> Component:
    r"""Returns 45deg crossing with bends.

    Args:
        crossing: crossing function.
        port_spacing: target I/O port spacing.
        dx: target length.
        alpha: optimization parameter. diminish it for tight bends,
          increase it if raises assertion angle errors
        npoints: number of points.
        cross_section: cross_section spec.
        cross_section_bends: cross_section spec.


    The 45 Degree crossing CANNOT be kept as an SRef since
    we only allow for multiples of 90Deg rotations in SRef.

    .. code::

        ----   ----
            \ /
             X
            / \
        ---    ----

    """
    crossing = gf.get_component(
        crossing, cross_section=cross_section_bends or cross_section
    )

    c = Component()
    x = c << crossing
    x.drotate(45)

    p_e = x.ports["o3"].dcenter
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
        cross_section=cross_section_bends,
    )

    tol = 1e-2
    assert abs(bend.info["start_angle"] - start_angle) < tol, print(
        f"{bend.info['start_angle']} differs from {start_angle}"
    )
    assert abs(bend.info["end_angle"] - end_angle) < tol, bend.info["end_angle"]

    b_tr = c << bend
    b_tl = c << bend
    b_bl = c << bend
    b_br = c << bend

    b_tr.connect("o2", x.ports["o3"], mirror=True)
    b_tl.connect("o2", x.ports["o1"], mirror=True)
    b_bl.connect("o2", x.ports["o4"])
    b_br.connect("o2", x.ports["o2"])

    c.info["bezier_length"] = bend.info["length"]
    c.info["min_bend_radius"] = bend.info["min_bend_radius"]

    c.add_port("o1", port=b_bl.ports["o1"])
    c.add_port("o2", port=b_tl.ports["o1"])
    c.add_port("o3", port=b_tr.ports["o1"])
    c.add_port("o4", port=b_br.ports["o1"])

    c.over_under(layer=bend.ports[0].layer)
    x = gf.get_cross_section(cross_section)
    x.add_bbox(c)
    return c


if __name__ == "__main__":
    c = crossing45()
    c.show()
