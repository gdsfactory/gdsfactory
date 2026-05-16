from __future__ import annotations

__all__ = ["bend_topic", "bend_topic180", "bend_topic_all_angle", "bend_topic_s"]

import warnings
from functools import partial
from typing import Literal, overload

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.path import topic
from gdsfactory.typings import AnyComponent, CrossSectionSpec, LayerSpec

from .._schematic import bend_schematic, sbend_schematic


@overload
def _bend_topic(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.1,
    npoints: int = 100,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    layer: LayerSpec | None = None,
    width: float | None = None,
    all_angle: Literal[False] = False,
) -> Component: ...


@overload
def _bend_topic(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.1,
    npoints: int = 100,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    layer: LayerSpec | None = None,
    width: float | None = None,
    all_angle: Literal[True] = True,
) -> ComponentAllAngle: ...


def _bend_topic(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.1,
    npoints: int = 100,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    layer: LayerSpec | None = None,
    width: float | None = None,
    all_angle: bool = False,
) -> AnyComponent:
    """Returns a Third Order Polynomial Interconnected Circular (TOPIC) bend component, as described in this publication https://arxiv.org/html/2411.15025v1.

    The bend consists of three parts:
    a. Initial transition from straight to bend, known as TOP segment.
    b. Circular part whose center and radius are calculated analytically.
    c. Mirroring of TOP segment with respect to the bisection of the angle.

    Args:
        radius: radius at the start and end of bend.
        angle: total angle of the curve in degrees.
        p: used to calculate the angle of the bend at the end of TOP / start of circular arc, as p*angle. It should be within [0, 0.5).
        npoints: Number of points used per 360 degrees.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        all_angle: if True, use all-angle extrusion/component handling for the TOPIC bend.

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
        raise ValueError("radius must be specified")

    if layer and width:
        x = gf.get_cross_section(cross_section, layer=layer, width=width)
    elif layer:
        x = gf.get_cross_section(cross_section, layer=layer)
    elif width:
        x = gf.get_cross_section(cross_section, width=width)

    path = topic(
        radius=radius,
        angle=angle,
        p=p,
        npoints=npoints,
    )

    c = path.extrude(x, all_angle=all_angle)

    min_bend_radius = float(np.round(path.info["Rmin"], 3))
    c.info["min_bend_radius"] = float(min_bend_radius)

    c.info["radius"] = radius
    c.info["length"] = float(np.round(path.length(), 3))
    c.info["dy"] = float(
        np.round(abs(float(path.points[0][1] - path.points[-1][1])), 3)
    )
    c.info["width"] = float(width or x.width)

    if not allow_min_radius_violation:
        x.validate_radius(min_bend_radius)

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


@gf.cell_with_module_name(schematic_function=bend_schematic, tags=["bends"])
def bend_topic(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.1,
    npoints: int = 100,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    layer: LayerSpec | None = None,
    width: float | None = None,
) -> Component:
    """Returns a regular degree Third Order Polynomial Interconnected Circular (TOPIC) bend component.

    The implementation follows the description in this publication https://arxiv.org/html/2411.15025v1.

    The bend consists of three parts:
    a. Initial transition from straight to bend, known as TOP segment.
    b. Circular part whose center and radius are calculated analytically.
    c. Mirroring of TOP segment with respect to the bisection of the angle.

    Args:
        radius: radius at the start and end of bend.
        angle: total angle of the curve in degrees.
        p: used to calculate the angle of the bend at the end of TOP / start of circular arc, as p*angle. It should be within [0, 0.5).
        npoints: Number of points used per 360 degrees.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
    """
    if abs(angle) not in {90, 180}:
        warnings.warn(
            f"bend_topic angle should be 90 or 180. Got {angle}. Use bend_topic_all_angle instead.",
            UserWarning,
            stacklevel=3,
        )
    return _bend_topic(
        radius=radius,
        angle=angle,
        p=p,
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        layer=layer,
        width=width,
        all_angle=False,
    )


@gf.vcell
def bend_topic_all_angle(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.1,
    npoints: int = 100,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    layer: LayerSpec | None = None,
    width: float | None = None,
) -> ComponentAllAngle:
    """Returns a Third Order Polynomial Interconnected Circular (TOPIC) bend component of arbitrary angle.

    The implementation follows the description in this publication https://arxiv.org/html/2411.15025v1.

    The bend consists of three parts:
    a. Initial transition from straight to bend, known as TOP segment.
    b. Circular part whose center and radius are calculated analytically.
    c. Mirroring of TOP segment with respect to the bisection of the angle.

    Args:
        radius: radius at the start and end of bend.
        angle: total angle of the curve in degrees.
        p: used to calculate the angle of the bend at the end of TOP / start of circular arc, as p*angle. It should be within [0, 0.5).
        npoints: Number of points used per 360 degrees.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
    """
    return _bend_topic(
        radius=radius,
        angle=angle,
        p=p,
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        layer=layer,
        width=width,
        all_angle=True,
    )


@gf.cell_with_module_name(schematic_function=sbend_schematic, tags=["bends"])
def bend_topic_s(
    radius: float | None = None,
    p: float = 0.1,
    npoints: int = 100,
    layer: LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    port1: str = "o1",
    port2: str = "o2",
) -> Component:
    r"""Sbend made of 2 topic bends.

    Args:
        radius: radius at the start and end of bend.
        p: used to calculate the angle of the bend at the end of TOP / start of circular arc, as p*angle. It should be within [0, 0.5).
        npoints: Number of points used per 360 degrees.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
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
    b = bend_topic(
        radius=radius,
        angle=90,
        p=p,
        npoints=npoints,
        layer=layer,
        width=width,
        allow_min_radius_violation=allow_min_radius_violation,
        cross_section=cross_section,
    )
    b1 = c.add_ref(b)
    b2 = c.add_ref(b)
    b2.connect(port1, b1[port2], mirror=True)
    c.add_port(port1, port=b1[port1])
    c.add_port(port2, port=b2[port2])
    c.info["length"] = 2 * b.info["length"]
    return c


bend_topic180 = partial(bend_topic, angle=180)
