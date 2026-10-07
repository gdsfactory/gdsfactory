import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.typings import CrossSectionSpec, LayerSpec

__all__ = [
    "bend_modified_hermite",
    "bend_modified_hermite180",
    "bend_modified_hermite_all_angle",
    "bend_modified_hermite_s",
]

from gdsfactory.component_functions import CellAlias

from .._schematic import bend_schematic, sbend_schematic


@gf.cell_with_module_name(schematic_function=bend_schematic, tags=["bends"])
def bend_modified_hermite(
    radius: float = 15,
    angle: float = 90.0,
    inner_tangent_magnitude: float = 26.5,
    outer_tangent_magnitude: float = 30,
    npoints: int = 100,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    layer: LayerSpec | None = None,
    width1: float | None = None,
    width2: float | None = None,
    port1: str = "o1",
    port2: str = "o2",
) -> Component:
    """Modified Hermite curve, described in "Low-Loss Silicon Nitride Bent Waveguides at O-Band with Modified Hermite Curves", Donghao Li et al, <https://www.mdpi.com/2304-6732/13/2/175> .

    Default parameters are taken from Table 3 of <https://www.mdpi.com/2304-6732/13/2/175> .

    Note that the default inner_tangent_magnitude and outer_tangent_magnitude parameters will need to be changed if you change radius or angle, as optimal values for those parameters depend on radius and angle.

    Args:
        radius: effective bend radius
        angle: angle, in degrees.
        inner_tangent_magnitude: a1 parameter from Li et al.
        outer_tangent_magnitude: a2 parameter from Li et al.
        npoints: number of points to use for the inner wall of the curve, and the outer wall.
        cross_section: spec (CrossSection, string or dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
        layer: layer to use. Defaults to cross_section.layer.
        width1: width to use at input. Defaults to cross_section.width.
        width2: width to use at output. Defaults to cross_section.width.
        port1: name of input port.
        port2: name of output port.
    """
    return cf.bend_modified_hermite(
        radius=radius,
        angle=angle,
        inner_tangent_magnitude=inner_tangent_magnitude,
        outer_tangent_magnitude=outer_tangent_magnitude,
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        layer=layer,
        width1=width1,
        width2=width2,
        port1=port1,
        port2=port2,
    )


@gf.vcell
def bend_modified_hermite_all_angle(
    radius: float = 15,
    angle: float = 90.0,
    inner_tangent_magnitude: float = 26.5,
    outer_tangent_magnitude: float = 30,
    npoints: int = 100,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    layer: LayerSpec | None = None,
    width1: float | None = None,
    width2: float | None = None,
    port1: str = "o1",
    port2: str = "o2",
) -> ComponentAllAngle:
    """Modified Hermite curve, described in "Low-Loss Silicon Nitride Bent Waveguides at O-Band with Modified Hermite Curves", Donghao Li et al, <https://www.mdpi.com/2304-6732/13/2/175> .

    Default parameters are taken from Table 3 of <https://www.mdpi.com/2304-6732/13/2/175> .

    This is the all_angle version that can handle angles that aren't integer multiples of 90 degrees.


    Note that the default inner_tangent_magnitude and outer_tangent_magnitude parameters will need to be changed if you change radius or angle, as optimal values for those parameters depend on radius and angle.

    Args:
        radius: effective bend radius
        angle: angle, in degrees.
        inner_tangent_magnitude: a1 parameter from Li et al.
        outer_tangent_magnitude: a2 parameter from Li et al.
        npoints: number of points to use for the inner wall of the curve, and the outer wall.
        cross_section: spec (CrossSection, string or dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
        layer: layer to use. Defaults to cross_section.layer.
        width1: width to use at input. Defaults to cross_section.width.
        width2: width to use at output. Defaults to cross_section.width.
        port1: name of input port.
        port2: name of output port.
    """
    return cf.bend_modified_hermite_all_angle(
        radius=radius,
        angle=angle,
        inner_tangent_magnitude=inner_tangent_magnitude,
        outer_tangent_magnitude=outer_tangent_magnitude,
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        layer=layer,
        width1=width1,
        width2=width2,
        port1=port1,
        port2=port2,
    )


@gf.cell_with_module_name(schematic_function=sbend_schematic, tags=["bends"])
def bend_modified_hermite_s(
    radius: float = 15,
    inner_tangent_magnitude: float = 26.5,
    outer_tangent_magnitude: float = 30,
    npoints: int = 100,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    layer: LayerSpec | None = None,
    width: float | None = None,
    port1: str = "o1",
    port2: str = "o2",
) -> Component:
    """Sbend made of 2 modified Hermite bends.

    Args:
        radius: effective bend radius
        inner_tangent_magnitude: a1 parameter from Li et al.
        outer_tangent_magnitude: a2 parameter from Li et al.
        npoints: number of points to use for the inner wall of the curve, and the outer wall.
        cross_section: spec (CrossSection, string or dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
        layer: layer to use. Defaults to cross_section.layer.
        width: width  at input and output (the width generally varies in the interior of the bend). Defaults to cross_section.width.
        port1: name of input port.
        port2: name of output port.
    """
    return cf.bend_modified_hermite_s(
        radius=radius,
        inner_tangent_magnitude=inner_tangent_magnitude,
        outer_tangent_magnitude=outer_tangent_magnitude,
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        layer=layer,
        width=width,
        port1=port1,
        port2=port2,
    )


bend_modified_hermite180 = CellAlias(bend_modified_hermite, angle=180)
