"""You can define a path with a list of points combined with a cross-section.

A path can be extruded using any CrossSection

Based on phidl.path
"""

from typing import Optional

from phidl.device_layout import CrossSection, Path

from pp import path as pa
from pp.component import Component
from pp.hash_points import hash_points
from pp.import_phidl_component import import_phidl_component
from pp.layers import LAYER
from pp.types import Number


def path(
    p: Path, cross_section: CrossSection, simplify: Optional[float] = None
) -> Component:
    """Returns an extruded Path into a Component

    Args:
        p: a path is a list of points (arc, straight, euler)
        cross_section: extrudes a cross_section over a path
        simplify: Tolerance value for the simplification algorithm.
            All points that can be removed without changing the resulting
            polygon by more than the value listed here will be removed.
    """
    device = p.extrude(cross_section=cross_section, simplify=simplify)
    component = import_phidl_component(component=device)
    component.name = f"path_{hash_points(p.points)}"
    return component


def arc(radius: Number = 10, angle: Number = 90, npoints: int = 720) -> Path:
    return pa.arc(radius=radius, angle=angle, num_pts=npoints)


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
    return pa.euler(radius=radius, angle=angle, p=p, use_eff=use_eff, num_pts=npoints)


def straight(length: Number = 10, npoints: int = 2) -> Path:
    return pa.straight(length=length, num_pts=npoints)


if __name__ == "__main__":
    import pp

    P = euler(radius=5)
    # P = euler()
    # P = Path()
    # P.append(straight(length=5))
    # P.append(pa.arc(radius=10, angle=90))
    # P.append(pa.spiral())

    # Create a blank CrossSection
    X = CrossSection()
    # X.add(width=2.0, offset=-4, layer=LAYER.HEATER, ports=["HW1", "HE1"])
    X.add(width=0.5, offset=0, layer=LAYER.SLAB90, ports=["in", "out"])
    # X.add(width=2.0, offset=4, layer=LAYER.HEATER, ports=["HW0", "HE0"])

    # Combine the Path and the CrossSection
    c = path(P, X)
    # c = pp.add_pins(c)
    c << pp.c.bend_euler90(radius=10)
    c << pp.c.bend_circular(radius=10)
    c.show()
