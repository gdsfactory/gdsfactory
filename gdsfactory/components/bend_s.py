from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory import cell
from gdsfactory.component import Component
from gdsfactory.components.bezier import bezier, bezier_curve
from gdsfactory.functions import curvature
from gdsfactory.typings import CrossSectionSpec


@cell
def bend_s(
    size: tuple[float, float] = (11.0, 1.8),
    npoints: int = 99,
    cross_section: CrossSectionSpec = "strip",
    allow_min_radius_violation: bool = False,
    **kwargs,
) -> Component:
    """Return S bend with bezier curve.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: in x and y direction.
        npoints: number of points.
        cross_section: spec.
        allow_min_radius_violation: bool.
        kwargs: cross_section settings.

    """
    dx, dy = size

    if dy == 0:
        return gf.components.straight(length=dx, cross_section=cross_section, **kwargs)

    return bezier(
        control_points=((0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy)),
        npoints=npoints,
        cross_section=cross_section,
        allow_min_radius_violation=allow_min_radius_violation,
        **kwargs,
    )


def get_min_sbend_size(
    size: tuple[float | None, float | None] = (None, 10.0),
    cross_section: CrossSectionSpec = "strip",
    num_points: int = 100,
    **kwargs,
) -> float:
    """Returns the minimum sbend size to comply with bend radius requirements.

    Args:
        size: in x and y direction. One of them is None, which is the size we need to figure out.
        cross_section: spec.
        num_points: number of points to iterate over between max_size and 0.1 * max_size.
        kwargs: cross_section settings.
    """
    size = list(size)
    cross_section_f = gf.get_cross_section(cross_section, **kwargs)

    if size[0] is None:
        ind = 0
        known_s = size[1]
    elif size[1] is None:
        ind = 1
        known_s = size[0]
    else:
        raise ValueError("One of the two elements in size has to be None")

    min_radius = cross_section_f.radius

    if min_radius is None:
        raise ValueError("The min radius for the specified layer is not known!")

    min_size = np.inf

    # Guess sizes, iterate over them until we cannot achieve the min radius
    # the max size corresponds to an ellipsoid
    max_size = 2.5 * np.sqrt(np.abs(min_radius * known_s))
    sizes = np.linspace(max_size, 0.1 * max_size, num_points)

    for s in sizes:
        sz = size
        sz[ind] = s
        dx, dy = size
        control_points = ((0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy))
        npoints = 201
        t = np.linspace(0, 1, npoints)
        path_points = bezier_curve(t, control_points)
        curv = curvature(path_points, t)
        min_bend_radius = 1 / max(np.abs(curv))
        if min_bend_radius < min_radius:
            min_size = s
            break

    return min_size


if __name__ == "__main__":
    min_size = get_min_sbend_size()
    print(min_size)
    # c = bend_s(size=(10, 0))
    # c = bend_s(bbox_offsets=[0.5], bbox_layers=[(111, 0)], width=2)
    # c = bend_s(size=[10, 2.5])  # 10um bend radius
    # c = bend_s(size=[20, 3], cross_section="rib")  # 10um bend radius
    # c.pprint()
    # c = bend_s_biased()
    # print(c.info["min_bend_radius"])
    # c.show()
