from functools import partial
import numpy as np
from typing import Literal, overload
import warnings

import gdsfactory as gf
from gdsfactory.component import Component, ComponentAllAngle
from gdsfactory.typings import AnyComponent, CrossSectionSpec, LayerSpec

__all__ = ["bend_modified_hermite", "bend_modified_hermite_all_angle", "bend_modified_hermite180", "bend_modified_hermite_s"]

from .._schematic import bend_schematic, sbend_schematic

# Hermite curve coefficients from https://www.mdpi.com/2304-6732/13/2/175 Eq. (1)
def _hermite_c1(t: float) -> float:
    return 1 - 3 * t**2 + 2 * t**3

def _hermite_c2(t: float) -> float:
    return t - 2 * t**2 + t**3

def _hermite_c3(t: float) -> float:
    return -t**2 + t**3

def _hermite_c4(t: float) -> float:
    return 3 * t**2 - 2 * t**3

# Hermite curve points from https://www.mdpi.com/2304-6732/13/2/175 Eq. (2)
def _hermite_curve(t: float, init_point: float, init_tangent: float, end_point: float, end_tangent: float) -> float:
    return _hermite_c1(t)[:, None] * init_point \
        + _hermite_c2(t)[:, None] * init_tangent \
        + _hermite_c3(t)[:, None] * end_tangent \
        + _hermite_c4(t)[:, None] * end_point

@overload
def _bend_modified_hermite(
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
    all_angle: Literal[False] = False,
) -> gf.Component: ...

@overload
def _bend_modified_hermite(
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
    all_angle: Literal[True] = True,
) -> gf.Component: ...

def _bend_modified_hermite(
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
    all_angle: bool=False,
) -> AnyComponent:
    """Modified Hermite curve, described in "Low-Loss Silicon Nitride Bent Waveguides at O-Band with Modified Hermite Curves", Donghao Li et al, https://www.mdpi.com/2304-6732/13/2/175
    
    Default parameters are taken from Table 3 of https://www.mdpi.com/2304-6732/13/2/175

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
        all_angle: if True, use all-angle extrusion/component handling for the bend.
    """

    xsec = gf.get_cross_section(cross_section)
    width1 = xsec.width if width1 is None else width1
    width2 = xsec.width if width2 is None else width2
    layer = xsec.layer if layer is None else layer

    end_point_unit_vector = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    end_tangent_unit_vector = np.array([-np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))])

    t = np.linspace(0, 1, npoints)

    # polygon points for inner bend
    inner_bend_init_point = np.array([radius - width1 / 2, 0])
    inner_bend_init_tangent = np.array([0, inner_tangent_magnitude])
    inner_bend_end_point = (radius - width2 / 2) * end_point_unit_vector
    inner_bend_end_tangent = inner_tangent_magnitude * end_tangent_unit_vector

    inner_bend_points = _hermite_curve(t=t,
                           init_point=inner_bend_init_point,
                           init_tangent=inner_bend_init_tangent,
                           end_point=inner_bend_end_point,
                           end_tangent=inner_bend_end_tangent)

    # polygon points for outer bend
    outer_bend_init_point = np.array([radius + width1 / 2, 0])
    outer_bend_init_tangent = np.array([0, outer_tangent_magnitude])
    outer_bend_end_point = (radius + width2 / 2) * end_point_unit_vector
    outer_bend_end_tangent = outer_tangent_magnitude * end_tangent_unit_vector

    outer_bend_points = _hermite_curve(t=t,
                       init_point=outer_bend_init_point,
                       init_tangent=outer_bend_init_tangent,
                       end_point=outer_bend_end_point,
                       end_tangent=outer_bend_end_tangent)

    # Calculate center of curve as average of inner and outer bend points
    interior_points = (inner_bend_points + outer_bend_points) / 2
    interior_path = gf.Path(interior_points)
    _, curvature = interior_path.curvature()
    min_bend_radius = np.min(1 / curvature)

    if xsec.radius_min is not None and not allow_min_radius_violation:
        xsec.validate_radius(radius=min_bend_radius)

    polygon_points = np.concat((inner_bend_points, np.flip(outer_bend_points, axis=0)), axis=0)

    result = gf.ComponentAllAngle() if all_angle else gf.Component()
    result.add_polygon(points=polygon_points, layer=layer)
    start_orientation = 90 if angle > 0 else -90
    result.add_port(name=port1, center=interior_points[0], width=width1, orientation=start_orientation, layer=layer)
    result.add_port(name=port2, center=interior_points[-1], width=width2, orientation=start_orientation + angle + 180, layer=layer)

    result.info['min_bend_radius'] = min_bend_radius
    result.info['length'] = interior_path.length()

    return result

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
    """Modified Hermite curve, described in "Low-Loss Silicon Nitride Bent Waveguides at O-Band with Modified Hermite Curves", Donghao Li et al, https://www.mdpi.com/2304-6732/13/2/175
    
    Default parameters are taken from Table 3 of https://www.mdpi.com/2304-6732/13/2/175

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
    """
    if angle not in {90, 180, 270}:
        warnings.warn(
            f"bend_euler angle should be 90 or 180. Got {angle}. Use bend_modified_hermite_all_angle instead.",
            UserWarning,
            stacklevel=3,
        )
    return _bend_modified_hermite(
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
        all_angle=False,
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
    """Modified Hermite curve, described in "Low-Loss Silicon Nitride Bent Waveguides at O-Band with Modified Hermite Curves", Donghao Li et al, https://www.mdpi.com/2304-6732/13/2/175
    
    Default parameters are taken from Table 3 of https://www.mdpi.com/2304-6732/13/2/175

    This is the all_angle version that can handle angles that aren't integer multiples of 90 degrees.

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
    """
    return _bend_modified_hermite(
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
        all_angle=True,
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
        angle: angle, in degrees.
        inner_tangent_magnitude: a1 parameter from Li et al.
        outer_tangent_magnitude: a2 parameter from Li et al.
        npoints: number of points to use for the inner wall of the curve, and the outer wall.
        cross_section: spec (CrossSection, string or dict).
        allow_min_radius_violation: if True allows radius to be smaller than cross_section radius.
        layer: layer to use. Defaults to cross_section.layer.
        width: width  at input and output (the width generally varies in the interior of the bend). Defaults to cross_section.width.
    """

    result = Component()
    bend = bend_modified_hermite(
        radius=radius,
        angle=90,
        inner_tangent_magnitude=inner_tangent_magnitude,
        outer_tangent_magnitude=outer_tangent_magnitude,
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        layer=layer,
        width1=width,
        width2=width,
        port1=port1,
        port2=port2,
    )
    bend1 = result << bend
    bend2 = result << bend
    bend2.connect(port1, bend1[port2], mirror=True)
    result.add_port(port1, port=bend1[port1])
    result.add_port(port2, port=bend2[port2])
    result.info['length'] = 2 * bend.info['length']
    return result

bend_modified_hermite180 = partial(bend_modified_hermite, angle=180)