from __future__ import annotations

__all__ = [
    "bend_s",
    "bend_s_offset",
    "bezier",
    "bezier_curve",
    "find_min_curv_bezier_control_points",
    "get_min_sbend_size",
]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions.bends.bend_s import (
    bezier_curve,
    find_min_curv_bezier_control_points,
    get_min_sbend_size,
)
from gdsfactory.config import ErrorType
from gdsfactory.typings import Coordinates, CrossSectionSpec, Size

from .._schematic import sbend_schematic


@gf.cell_with_module_name(schematic_function=sbend_schematic, tags=["bends"])
def bezier(
    control_points: Coordinates = ((0.0, 0.0), (5.0, 0.0), (5.0, 1.8), (10.0, 1.8)),
    npoints: int = 201,
    with_manhattan_facing_angles: bool = True,
    start_angle: int | None = None,
    end_angle: int | None = None,
    cross_section: CrossSectionSpec = "strip",
    bend_radius_error_type: ErrorType | None = None,
    allow_min_radius_violation: bool = False,
    width: float | None = None,
    width_function: gf.typings.WidthFunction | None = None,
) -> Component:
    """Returns Bezier bend.

    Args:
        control_points: list of points.
        npoints: number of points varying between 0 and 1.
        with_manhattan_facing_angles: bool.
        start_angle: optional start angle in deg.
        end_angle: optional end angle in deg.
        cross_section: spec.
        bend_radius_error_type: error type.
        allow_min_radius_violation: bool.
        width: width to use. Defaults to cross_section.width.
        width_function: optional main-strip width along the extruded path.
    """
    return cf.bezier(
        control_points=control_points,
        npoints=npoints,
        with_manhattan_facing_angles=with_manhattan_facing_angles,
        start_angle=start_angle,
        end_angle=end_angle,
        cross_section=cross_section,
        bend_radius_error_type=bend_radius_error_type,
        allow_min_radius_violation=allow_min_radius_violation,
        width=width,
        width_function=width_function,
    )


@gf.cell_with_module_name(schematic_function=sbend_schematic, tags=["bends"])
def bend_s(
    size: Size = (11.0, 1.8),
    npoints: int = 99,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    width: float | None = None,
) -> Component:
    """Return S bend with bezier curve.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: in x and y direction.
        npoints: number of points.
        cross_section: spec.
        allow_min_radius_violation: bool.
        width: width to use. Defaults to cross_section.width.

    """
    return cf.bend_s(
        size=size,
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        width=width,
    )


@gf.cell_with_module_name(schematic_function=sbend_schematic, tags=["bends"])
def bend_s_offset(
    offset: float = 40.0,
    radius: float | None = 10.0,
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
    with_euler: bool | None = None,
    p: float = 1,
    with_arc_floorplan: bool = False,
    npoints: int | None = None,
    angular_step: float | None = None,
) -> gf.Component:
    """Return S bend made of two bends with a straight section.

    Args:
        offset: in um.
        radius: in um. if None, uses cross_section_radius.
        cross_section: spec.
        width: width to use. Defaults to cross_section.width.
        with_euler: deprecated, use p=0 for circular arc instead.
        p: 1 means standard Euler bend. 0 means circular arc.
        with_arc_floorplan: if True the size of the bend will be adjusted to match an arc bend with the specified radius. If False: `radius` is the minimum radius of curvature.
        npoints: number of points.
        angular_step: If provided, determines the angular step (in degrees) between points. Mutually exclusive with npoints.
    """
    return cf.bend_s_offset(
        offset=offset,
        radius=radius,
        cross_section=cross_section,
        width=width,
        with_euler=with_euler,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        angular_step=angular_step,
    )
