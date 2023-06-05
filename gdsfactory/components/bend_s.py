from __future__ import annotations

import copy

import numpy as np

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bezier import bezier
from gdsfactory.config import logger
from gdsfactory.typings import CrossSectionSpec, Float2


@cell
def bend_s(
    size: Float2 = (11.0, 2.0),
    npoints: int = 99,
    cross_section: CrossSectionSpec = "strip",
    check_min_radius: bool = False,
    **kwargs,
) -> Component:
    """Return S bend with bezier curve.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: in x and y direction.
        npoints: number of points.
        cross_section: spec.
        check_min_radius: raise ValueError if radius below min_bend_radius.
        kwargs: cross_section settings.
    """
    c = Component()
    dx, dy = size

    bend = bezier(
        control_points=((0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy)),
        npoints=npoints,
        cross_section=cross_section,
        **kwargs,
    )
    bend_ref = c << bend
    c.add_ports(bend_ref.ports)
    c.copy_child_info(bend)

    cross_section = gf.get_cross_section(cross_section, **kwargs)
    if (
        cross_section.radius is not None
        and c.info["min_bend_radius"] < cross_section.radius
    ):
        if check_min_radius:
            raise ValueError(
                f"The min bend radius of the generated s bend {c.info['min_bend_radius']} "
                f"is below the bend radius of the waveguide {cross_section.radius}"
            )
        else:
            logger.warning(
                f"The min bend radius of the generated s bend {c.info['min_bend_radius']} is below the bend radius of the waveguide {cross_section.radius}"
            )

    return c


def get_min_sbend_size(
    size: Float2 = (None, 10.0),
    cross_section: CrossSectionSpec = "strip",
    num_points: int = 100,
    **kwargs,
):
    """
    Returns the minimum sbend size to comply with bend radius requirements.

     Args:
        size: in x and y direction. One of them is None, which is the size we need to figure out.
        cross_section: spec.
        num_points: number of points to iterate over between max_size and 0.1 * max_size
        kwargs: cross_section settings.

    """

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

    for i, s in enumerate(sizes):
        sz = copy.deepcopy(size)
        sz[ind] = s
        # print(sz)
        try:
            bend_s(
                size=sz, cross_section=cross_section, check_min_radius=True, **kwargs
            )
            # print(c.info['min_bend_radius'])
            min_size = sizes[i]
        except ValueError:
            break

    return min_size


if __name__ == "__main__":
    c = bend_s(bbox_offsets=[0.5], bbox_layers=[(111, 0)], width=2)
    # c = bend_s(size=[10, 2.5])  # 10um bend radius
    # c = bend_s(size=[20, 3], cross_section="rib")  # 10um bend radius
    # c.pprint()
    # c = bend_s_biased()
    # print(c.info["min_bend_radius"])
    c.show(show_ports=False)
