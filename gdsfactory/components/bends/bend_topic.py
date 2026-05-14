from typing import overload

from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.typings import AnyComponent, CrossSectionSpec, LayerSpec

__all__ = ["bend_topic"]

import gdsfactory as gf
from gdsfactory.path import topic

from .._schematic import bend_schematic


@overload
def _bend_topic(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.1,
    npoints: float = 100,
    cross_section: CrossSectionSpec = "strip",
    layer: LayerSpec | None = None,
    width: float | None = None,
    all_angle: bool = False,
) -> Component: ...


@overload
def _bend_topic(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.1,
    npoints: float = 100,
    cross_section: CrossSectionSpec = "strip",
    layer: LayerSpec | None = None,
    width: float | None = None,
    all_angle: bool = False,
) -> ComponentAllAngle: ...


def _bend_topic(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.1,
    npoints: float = 100,
    cross_section: CrossSectionSpec = "strip",
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
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
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
        raise ValueError("radius must be specified")

    if layer and width:
        x = gf.get_cross_section(
            cross_section, layer=layer or x.layer, width=width or x.width
        )
    elif layer:
        x = gf.get_cross_section(cross_section, layer=layer or x.layer)
    elif width:
        x = gf.get_cross_section(cross_section, width=width or x.width)

    path = topic(
        radius=radius,
        angle=angle,
        p=p,
        npoints=npoints,
    )

    c = path.extrude(x, all_angle=all_angle)

    return c


@gf.cell_with_module_name(schematic_function=bend_schematic, tags=["bends"])
def bend_topic(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.1,
    npoints: float = 100,
    cross_section: CrossSectionSpec = "strip",
    layer: LayerSpec | None = None,
    width: float | None = None,
    all_angle: bool = False,
) -> Component:
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
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        all_angle: if True, the bend is drawn with a single euler curve.
    """
    return _bend_topic(
        radius=radius,
        angle=angle,
        p=p,
        npoints=npoints,
        cross_section=cross_section,
        layer=layer,
        width=width,
        all_angle=False,
    )
