from typing import Iterable, Optional

import pp
from pp.add_padding import get_padding_points
from pp.component import Component
from pp.components.bezier import bezier
from pp.cross_section import strip
from pp.tech import TECH
from pp.types import CrossSectionFactory, Layer


@pp.cell
def bend_s(
    height: float = 2.0,
    length: float = 10.0,
    nb_points: int = 99,
    layers_cladding: Optional[Iterable[Layer]] = TECH.waveguide.strip.layers_cladding,
    cladding_offset: float = TECH.waveguide.strip.cladding_offset,
    cross_section_factory: Optional[CrossSectionFactory] = None,
    **cross_section_settings,
) -> Component:
    """S bend with bezier curve

    Args:
        height: in y direction
        length: in x direction
        layer: gds number
        nb_points: number of points
        cross_section_factory
        **cross_section_settings

    .. plot::
      :include-source:

      import pp

      c = pp.components.bend_s(height=20)
      c.plot()

    """
    l, h = length, height
    cross_section_factory = cross_section_factory or strip
    cross_section = cross_section_factory(**cross_section_settings)
    width = cross_section.info["width"]
    layer = cross_section.info["layer"]

    c = bezier(
        width=width,
        control_points=[(0, 0), (l / 2, 0), (l / 2, h), (l, h)],
        npoints=nb_points,
        layer=layer,
    )
    c.add_port(name="W0", port=c.ports.pop("0"))
    c.add_port(name="E0", port=c.ports.pop("1"))

    points = get_padding_points(
        component=c, default=0, bottom=cladding_offset, right=0, top=cladding_offset
    )
    for layer in layers_cladding or []:
        c.add_polygon(points, layer=layer)

    return c


if __name__ == "__main__":
    c = bend_s(width=1)
    c.pprint()
    # c = bend_s_biased()
    # print(c.info["min_bend_radius"])
    c.show()
