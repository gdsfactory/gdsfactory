from __future__ import annotations

import warnings
from functools import partial
from typing import Literal, overload

import gdsfactory as gf
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.path import arc
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import CrossSectionSpec, LayerSpec


@overload
def _bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    npoints: int | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    all_angle: Literal[False] = False,
) -> Component: ...


@overload
def _bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    npoints: int | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    all_angle: Literal[True] = True,
) -> ComponentAllAngle: ...


def _bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    npoints: int | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    all_angle: bool = False,
) -> Component | ComponentAllAngle:
    """Returns a radial arc.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: angle of arc (degrees).
        npoints: number of points.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: spec (CrossSection, string or dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
        all_angle: if True returns a ComponentAllAngle.

    .. code::

                  o2
                  |
                 /
                /
        o1_____/
    """
    x = gf.get_cross_section(cross_section)
    radius = radius or x.radius
    assert radius is not None
    if layer and width:
        x = gf.get_cross_section(
            cross_section, layer=layer or x.layer, width=width or x.width
        )
    elif layer:
        x = gf.get_cross_section(cross_section, layer=layer or x.layer)
    elif width:
        x = gf.get_cross_section(cross_section, width=width or x.width)

    p = arc(radius=radius, angle=angle, npoints=npoints)
    c = p.extrude(x, all_angle=all_angle)

    c.info["length"] = float(snap_to_grid(p.length()))
    c.info["dy"] = float(abs(p.points[0][0] - p.points[-1][0]))
    c.info["radius"] = float(radius)
    c.info["width"] = width or x.width
    top = None if int(angle) in {180, -180, -90} else 0
    bottom = 0 if int(angle) in {-90} else None
    x.add_bbox(c, top=top, bottom=bottom)
    if not allow_min_radius_violation:
        x.validate_radius(radius)
    c.add_route_info(
        cross_section=x,
        length=c.info["length"],
        n_bend_90=abs(angle / 90.0),
        min_bend_radius=radius,
    )
    return c


@gf.cell
def bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    npoints: int | None = None,
    layer: gf.typings.LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
) -> Component:
    """Returns a radial arc.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: angle of arc (degrees).
        npoints: number of points.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: spec (CrossSection, string or dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
    """
    if angle not in {90, 180}:
        warnings.warn(
            f"bend_euler angle should be 90 or 180. Got {angle}. Use bend_euler_all_angle instead.",
            UserWarning,
            stacklevel=2,
        )
    return _bend_circular(
        radius=radius,
        angle=angle,
        npoints=npoints,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        all_angle=False,
    )


@gf.vcell
def bend_circular_all_angle(
    radius: float | None = None,
    angle: float = 90.0,
    npoints: int | None = None,
    layer: gf.typings.LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
) -> ComponentAllAngle:
    """Returns a radial arc.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: angle of arc (degrees).
        npoints: number of points.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: spec (CrossSection, string or dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
    """
    return _bend_circular(
        radius=radius,
        angle=angle,
        npoints=npoints,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        all_angle=True,
    )


bend_circular180 = partial(bend_circular, angle=180)


if __name__ == "__main__":
    c = gf.Component()
    r = c << bend_circular(radius=5)
    bend = bend_circular_all_angle(radius=5)
    print(type(bend))
    # r.dmirror()
    c.show()
