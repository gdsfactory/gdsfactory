"""Waveguide crossings."""

from __future__ import annotations

import numpy as np
from kfactory.conf import CheckInstances

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bends.bend_s import (
    bezier,
    find_min_curv_bezier_control_points,
)
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Delta, LayerSpec


@gf.cell_with_module_name
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
    c << gf.c.ellipse(radii=(r1, r2), layer=layer_slab)

    xs = gf.get_cross_section(cross_section)
    width = xs.width
    assert xs.layer is not None
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


@gf.cell_with_module_name
def crossing(
    arm: ComponentSpec = crossing_arm,
) -> gf.Component:
    """Waveguide crossing.

    Args:
        arm: arm spec.
    """
    c = gf.Component()
    arm = gf.get_component(arm)
    for rotation in [0, 90, 180, 270]:
        ref = c << arm
        ref.rotate(rotation)
        c.add_port(port=ref["o2"])
    c.auto_rename_ports()
    c.flatten()
    return c


@gf.cell_with_module_name
def crossing_linear_taper(
    width1: float = 2.5,
    width2: float = 0.5,
    length: float = 3,
    cross_section: CrossSectionSpec = "strip",
    taper: ComponentSpec = "taper",
) -> Component:
    """Returns Crossing based on a taper.

    The default is a dummy taper.

    Args:
        width1: input width.
        width2: output width.
        length: taper length.
        cross_section: cross_section spec.
        taper: taper spec.
    """
    arm = gf.get_component(
        taper, width1=width1, width2=width2, length=length, cross_section=cross_section
    )
    return crossing(arm=arm)


@gf.cell_with_module_name
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
    _ = c << gf.c.ellipse(radii=(r1, r2), layer=layer_wg)
    _ = c << gf.c.ellipse(radii=(r2, r1), layer=layer_wg)

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


@gf.cell(check_instances=CheckInstances.IGNORE, with_module_name=True)
def crossing45(
    crossing: ComponentSpec = crossing,
    port_spacing: float = 40.0,
    dx: Delta | None = None,
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
    crossing = gf.get_component(crossing)

    c = Component()
    x = c << crossing
    x.rotate(45)

    p_e = x.ports["o3"].center
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
    assert abs(bend.info["start_angle"] - start_angle) < tol, (
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

    xs = gf.get_cross_section(cross_section)
    xs.add_bbox(c)
    return c


__all__ = [
    "crossing",
    "crossing45",
    "crossing_etched",
    "crossing_linear_taper",
]

if __name__ == "__main__":
    c = crossing45()
    c.show()
