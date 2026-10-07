from __future__ import annotations

__all__ = ["bend_circular", "bend_circular180", "bend_circular_all_angle"]

from typing import Unpack

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import CrossSectionSpec, ExtrusionPorts

from .._schematic import bend_schematic


@gf.cell_with_module_name(schematic_function=bend_schematic, tags=["bends"])
def bend_circular(
    radius: float | None = None,
    angle: float = 90.0,
    npoints: int | None = None,
    angular_step: float | None = None,
    layer: gf.typings.LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    **kwargs: Unpack[ExtrusionPorts],
) -> Component:
    """Returns a radial arc.

    Args:
        radius: in um. Defaults to cross_section_radius.
        angle: angle of arc (degrees).
        npoints: number of points.
        angular_step: If provided, determines the angular step (in degrees) between points. Mutually exclusive with npoints.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: spec (CrossSection, string or dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
        kwargs: optional ``port_type`` override for ports o1/o2.
            Defaults to the PDK's port policy.
    """
    return cf.bend_circular(
        radius=radius,
        angle=angle,
        npoints=npoints,
        angular_step=angular_step,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        **kwargs,
    )


@gf.vcell
def bend_circular_all_angle(
    radius: float | None = None,
    angle: float = 90.0,
    npoints: int | None = None,
    angular_step: float | None = None,
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
        angular_step: If provided, determines the angular step (in degrees) between points. Mutually exclusive with npoints.
        layer: layer to use. Defaults to cross_section.layer.
        width: width to use. Defaults to cross_section.width.
        cross_section: spec (CrossSection, string or dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
    """
    return cf.bend_circular_all_angle(
        radius=radius,
        angle=angle,
        npoints=npoints,
        angular_step=angular_step,
        layer=layer,
        width=width,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
    )


bend_circular180 = CellAlias(bend_circular, angle=180)
