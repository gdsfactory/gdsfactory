from __future__ import annotations

__all__ = ["bend_topic", "bend_topic180", "bend_topic_all_angle", "bend_topic_s"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from .._schematic import bend_schematic, sbend_schematic


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

    The implementation follows the description in this publication <https://arxiv.org/html/2411.15025v1>.

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
    return cf.bend_topic(
        radius=radius,
        angle=angle,
        p=p,
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        layer=layer,
        width=width,
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

    The implementation follows the description in this publication <https://arxiv.org/html/2411.15025v1>.

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
    return cf.bend_topic_all_angle(
        radius=radius,
        angle=angle,
        p=p,
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        layer=layer,
        width=width,
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

    ```text
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
    ```

    """
    return cf.bend_topic_s(
        radius=radius,
        p=p,
        npoints=npoints,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        port1=port1,
        port2=port2,
    )


bend_topic180 = CellAlias(bend_topic, angle=180)
