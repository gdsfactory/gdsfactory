from __future__ import annotations

import warnings
from functools import partial
from typing import Literal, overload

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.path import euler
from gdsfactory.typings import AnyComponent, CrossSectionSpec, LayerSpec


@overload
def _bend_euler(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    all_angle: Literal[False] = False,
) -> Component: ...


@overload
def _bend_euler(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    all_angle: Literal[True] = True,
) -> ComponentAllAngle: ...


def _bend_euler(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    all_angle: bool = False,
) -> AnyComponent:
    """Euler bend with changing bend radius.

    By default, `radius` corresponds to the minimum radius of curvature of the bend.
    However, if `with_arc_floorplan` is True, `radius` corresponds to the effective
    radius of curvature (making the curve a drop-in replacement for an arc). If
    p < 1.0, will create a "partial euler" curve as described in Vogelbacher et.
    al. https://dx.doi.org/10.1364/oe.27.031394

    default p = 0.5 based on this paper
    https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-25-8-9150&id=362937

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: total angle of the curve.
        p: Proportion of the curve that is an Euler curve.
        with_arc_floorplan: If False: `radius` is the minimum radius of curvature
          If True: The curve scales such that the endpoints match a bend_circular
          with parameters `radius` and `angle`.
        npoints: Number of points used per 360 degrees.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
        all_angle: if True, the bend is drawn with a single euler curve.

    .. code::

                  o2
                  |
                 /
                /
               /
       o1_____/
    """
    x = gf.get_cross_section(cross_section)
    radius = radius or x.radius

    if radius is None:
        return gf.c.wire_corner(cross_section=x)

    if layer and width:
        x = gf.get_cross_section(
            cross_section, layer=layer or x.layer, width=width or x.width
        )
    elif layer:
        x = gf.get_cross_section(cross_section, layer=layer or x.layer)
    elif width:
        x = gf.get_cross_section(cross_section, width=width or x.width)

    path = euler(
        radius=radius, angle=angle, p=p, use_eff=with_arc_floorplan, npoints=npoints
    )
    c = path.extrude(x, all_angle=all_angle)
    min_bend_radius = float(np.round(path.info["Rmin"], 3))
    c.info["length"] = float(np.round(path.length(), 3))
    c.info["dy"] = float(
        np.round(abs(float(path.points[0][0] - path.points[-1][0])), 3)
    )
    c.info["min_bend_radius"] = min_bend_radius
    c.info["radius"] = float(radius)
    c.info["width"] = width or x.width

    if not allow_min_radius_violation:
        x.validate_radius(radius)

    top = None if int(angle) in {180, -180, -90} else 0
    bottom = 0 if int(angle) in {-90} else None
    x.add_bbox(c, top=top, bottom=bottom)
    c.add_route_info(
        cross_section=x,
        length=c.info["length"],
        n_bend_90=abs(angle / 90.0),
        min_bend_radius=min_bend_radius,
    )
    return c


@gf.cell
def bend_euler_s(
    radius: float | None = None,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    port1: str = "o1",
    port2: str = "o2",
) -> Component:
    r"""Sbend made of 2 euler bends.

    Args:
        radius: in um. Defaults to cross_section_radius.
        p: Proportion of the curve that is an Euler curve.
        with_arc_floorplan: If False: `radius` is the minimum radius of curvature.
        npoints: Number of points used per 360 degrees.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
        port1: input port name.
        port2: output port name.


    .. code::

                        _____ o2
                       /
                      /
                     /
                    /
                    |
                   /
                  /
                 /
         o1_____/

    """
    c = Component()
    b = bend_euler(
        radius=radius,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
    )
    b1 = c.add_ref(b)
    b2 = c.add_ref(b)
    b2.connect(port1, b1[port2], mirror=True)
    b2.connect(port1, b1[port2])
    c.add_port(port1, port=b1[port1])
    c.add_port(port2, port=b2[port2])
    c.info["length"] = 2 * b.info["length"]
    return c


@gf.cell
def bend_euler(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
) -> Component:
    """Regular degree euler bend.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: total angle of the curve.
        p: Proportion of the curve that is an Euler curve.
        with_arc_floorplan: if True the size of the bend will be adjusted to match an arc bend with the specified radius. If False: `radius` is the minimum radius of curvature.
        npoints: Number of points used per 360 degrees.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
    """
    if abs(angle) not in {90, 180}:
        warnings.warn(
            f"bend_euler angle should be 90 or 180. Got {angle}. Use bend_euler_all_angle instead.",
            UserWarning,
            stacklevel=2,
        )

    return _bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        all_angle=False,
    )


@gf.vcell
def bend_euler_all_angle(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int | None = None,
    layer: gf.typings.LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
) -> ComponentAllAngle:
    """Regular degree euler bend.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: total angle of the curve.
        p: Proportion of the curve that is an Euler curve.
        with_arc_floorplan: If False: `radius` is the minimum radius of curvature
        npoints: Number of points used per 360 degrees.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.

    """
    return _bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        all_angle=True,
    )


bend_euler180 = partial(bend_euler, angle=180)


if __name__ == "__main__":
    # c = bend_euler(cross_section="rib", angle=90, radius=5)
    # c = bend_euler(cross_section="rib", angle=90, radius=20, clockwise=True)
    c = bend_euler(angle=90)
    c = gf.grid(
        [
            bend_euler(p=0, with_arc_floorplan=False),
            bend_euler(p=1, with_arc_floorplan=True),
        ]
    )
    c.show()
