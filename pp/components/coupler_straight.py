from phidl.device_layout import CrossSection

from pp.cell import cell
from pp.component import Component
from pp.path import component, straight
from pp.tech import TECH_SILICON_C, Tech


@cell
def coupler_straight(
    length: float = 10.0,
    gap: float = 0.27,
    tech: Tech = TECH_SILICON_C,
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
    x = CrossSection()
    x.add(width=tech.wg_width, offset=0, layer=tech.layer_wg, ports=["W0", "E0"])
    x.add(
        width=tech.wg_width,
        offset=gap + tech.wg_width,
        layer=tech.layer_wg,
        ports=["W1", "E1"],
    )
    for layer_cladding in tech.layers_cladding:
        x.add(
            width=tech.wg_width + 2 * tech.cladding_offset,
            offset=0,
            layer=layer_cladding,
        )

    p = straight(length=length)
    return component(p, x)


if __name__ == "__main__":
    c = coupler_straight()
    c.show()
