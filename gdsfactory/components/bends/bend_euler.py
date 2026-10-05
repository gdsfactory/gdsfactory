from __future__ import annotations

__all__ = ["bend_euler", "bend_euler180", "bend_euler_all_angle", "bend_euler_s"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from .._schematic import bend_schematic, sbend_schematic


@gf.cell_with_module_name(schematic_function=sbend_schematic, tags=["bends"])
def bend_euler_s(
    radius: float | None = None,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int | None = None,
    angular_step: float | None = None,
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
        angular_step: if not None, the angle step in degrees for the all_angle bend.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
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
    return cf.bend_euler_s(
        radius=radius,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        angular_step=angular_step,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        port1=port1,
        port2=port2,
    )


@gf.cell_with_module_name(schematic_function=bend_schematic, tags=["bends"])
def bend_euler(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int | None = None,
    angular_step: float | None = None,
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
        angular_step: if not None, the angle step in degrees for the all_angle bend.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
    """
    return cf.bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        angular_step=angular_step,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
    )


@gf.vcell
def bend_euler_all_angle(
    radius: float | None = None,
    angle: float = 90.0,
    p: float = 0.5,
    with_arc_floorplan: bool = True,
    npoints: int | None = None,
    angular_step: float | None = None,
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
        angular_step: if not None, the angle step in degrees for the all_angle bend.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.

    """
    return cf.bend_euler_all_angle(
        radius=radius,
        angle=angle,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        angular_step=angular_step,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
    )


bend_euler180 = CellAlias(bend_euler, angle=180)
