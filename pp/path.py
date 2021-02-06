"""You can define a path with a list of points combined with a cross-section.

A path can be extruded using any CrossSection returning a Component

The CrossSection defines the layer numbers, widths and offsetts

Based on phidl.path
"""

from collections.abc import Iterable
from typing import Optional

import numpy as np
import phidl.path as path
from phidl.device_layout import CrossSection, Path
from phidl.path import transition

from pp.component import Component
from pp.hash_points import hash_points
from pp.import_phidl_component import import_phidl_component
from pp.layers import LAYER
from pp.port import auto_rename_ports
from pp.types import Number


def component(
    p: Path,
    cross_section: CrossSection,
    simplify: Optional[float] = None,
    rename_ports: bool = True,
) -> Component:
    """Returns Component extruding a Path with a cross_section.

    A path can be extruded using any CrossSection returning a Component

    The CrossSection defines the layer numbers, widths and offsetts

    Args:
        p: a path is a list of points (arc, straight, euler)
        cross_section: extrudes a cross_section over a path
        simplify: Tolerance value for the simplification algorithm.
            All points that can be removed without changing the resulting
            polygon by more than the value listed here will be removed.
    """
    device = p.extrude(cross_section=cross_section, simplify=simplify)
    c = import_phidl_component(component=device)

    xsection_points = []
    for s in cross_section.sections:
        width = s["width"]
        offset = s["offset"]
        layer = s["layer"]
        if isinstance(offset, int) and isinstance(width, int):
            xsection_points.append([width, offset])
        if isinstance(layer, int):
            xsection_points.append([layer, 0])
        elif (
            isinstance(layer, Iterable)
            and len(layer) > 1
            and isinstance(layer[0], int)
            and isinstance(layer[1], int)
        ):
            xsection_points.append([layer[0], layer[1]])

    points = np.concatenate((p.points, np.array(xsection_points)))
    c.name = f"path_{hash_points(points)}"
    if rename_ports:
        auto_rename_ports(c)
    return c


def arc(radius: Number = 10, angle: Number = 90, npoints: int = 720) -> Path:
    return path.arc(radius=radius, angle=angle, num_pts=npoints)


def euler(
    radius: Number = 10,
    angle: Number = 90,
    p: float = 1,
    use_eff: bool = False,
    npoints: int = 720,
) -> Path:
    """Returns an euler bend that adiabatically transitions from straight to curved.
    By default, `radius` corresponds to the minimum radius of curvature of the bend.
    However, if `use_eff` is set to True, `radius` corresponds to the effective
    radius of curvature (making the curve a drop-in replacement for an arc). If
    p < 1.0, will create a "partial euler" curve as described in Vogelbacher et.
    al. https://dx.doi.org/10.1364/oe.27.031394

    Args:
        radius: minimum radius of curvature
        angle: total angle of the curve
        p: Proportion of the curve that is an Euler curve
        use_eff: If False: `radius` is the minimum radius of curvature of the bend
            If True: The curve will be scaled such that the endpoints match an arc
            with parameters `radius` and `angle`
        num_pts: Number of points used per 360 degrees

    """
    return path.euler(radius=radius, angle=angle, p=p, use_eff=use_eff, num_pts=npoints)


def straight(length: Number = 10, npoints: int = 100) -> Path:
    """Returns a straight path

    For transitions you should increase have at least 100 points
    """
    return path.straight(length=length, num_pts=npoints)


__all__ = ["straight", "euler", "arc", "component", "path", "transition"]

if __name__ == "__main__":
    import pp

    P = euler(radius=10, use_eff=True)
    # P = euler()
    # P = Path()
    # P.append(straight(length=5))
    # P.append(path.arc(radius=10, angle=90))
    # P.append(path.spiral())

    # Create a blank CrossSection
    X = CrossSection()
    # X.add(width=2.0, offset=-4, layer=LAYER.HEATER, ports=["HW1", "HE1"])
    X.add(width=0.5, offset=0, layer=LAYER.SLAB90, ports=["in", "out"])
    # X.add(width=2.0, offset=4, layer=LAYER.HEATER, ports=["HW0", "HE0"])

    # Combine the Path and the CrossSection
    c = component(P, X)
    # c = pp.add_pins(c)
    # c << pp.c.bend_euler(radius=10)
    c << pp.c.bend_circular(radius=10)
    c.show()
