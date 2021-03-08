from typing import Iterable, Optional

from phidl.device_layout import CrossSection

from pp.cell import cell
from pp.component import Component
from pp.path import component, straight
from pp.tech import TECH_SILICON_C, Tech
from pp.types import Layer


@cell
def coupler_straight(
    length: float = 10.0,
    gap: float = 0.27,
    width: float = TECH_SILICON_C.wg_width,
    layer: Layer = TECH_SILICON_C.layer_wg,
    layers_cladding: Optional[Iterable[Layer]] = None,
    cladding_offset: Optional[float] = None,
    tech: Optional[Tech] = None,
) -> Component:
    """Coupler_straight with two parallel waveguides.

    Args:
        length: of straight
        gap: between waveguides
        tech: Technology

    .. plot::
      :include-source:

      import pp

      c = pp.c.waveguide(length=10)
      c.plot()

    """
    tech = tech or TECH_SILICON_C
    cladding_offset = cladding_offset or getattr(tech, "cladding_offset", 0)
    layers_cladding = layers_cladding or getattr(tech, "layers_cladding", [])

    cross_section = CrossSection()
    cross_section.add(width=width, offset=0, layer=layer, ports=["W0", "E0"])
    cross_section.add(
        width=width,
        offset=gap + width,
        layer=layer,
        ports=["W1", "E1"],
    )
    layers_cladding = layers_cladding or []
    for layer_cladding in layers_cladding:
        cross_section.add(
            width=width + 2 * cladding_offset,
            offset=0,
            layer=layer_cladding,
        )

    p = straight(length=length)
    c = component(p, cross_section, snap_to_grid_nm=tech.snap_to_grid_nm)
    return c


if __name__ == "__main__":
    c = coupler_straight()
    c.show()
