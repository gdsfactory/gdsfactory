import pp
from pp.components.bend_s import bend_s
from pp.component import Component
from typing import Callable, List, Tuple


@pp.autoname
def coupler_symmetric(
    bend: Callable = bend_s,
    gap: float = 0.234,
    wg_width: float = 0.5,
    layer: Tuple[int, int] = pp.LAYER.WG,
    layers_cladding: List[Tuple[int, int]] = [pp.LAYER.WGCLAD],
    cladding_offset: float = 3.0,
) -> Component:
    """ two coupled waveguides with bends

    Args:
        bend:
        gap: um

    .. plot::
      :include-source:

      import pp

      c = pp.c.coupler_symmetric()
      pp.plotgds(c)

    """
    bend = pp.call_if_func(
        bend,
        width=wg_width,
        layer=layer,
        layers_cladding=layers_cladding,
        cladding_offset=cladding_offset,
        pins=False,
    )

    w = bend.ports["W0"].width
    y = (w + gap) / 2

    c = pp.Component()
    top_bend = bend.ref(position=(0, y), port_id="W0")
    bottom_bend = bend.ref(position=(0, -y), port_id="W0", v_mirror=True)

    c.add(top_bend)
    c.add(bottom_bend)

    # Using absorb here to have a flat cell and avoid
    # to have deeper hierarchy than needed
    c.absorb(top_bend)
    c.absorb(bottom_bend)

    port_width = 2 * w + gap
    c.add_port(name="W0", midpoint=[0, 0], width=port_width, orientation=180)
    c.add_port(port=bottom_bend.ports["E0"], name="E0")
    c.add_port(port=top_bend.ports["E0"], name="E1")

    return c


@pp.autoname
def coupler_symmetric_biased(bend=bend_s, gap=0.2, wg_width=0.5):
    return coupler_symmetric(
        bend=bend, gap=pp.bias.gap(gap), wg_width=pp.bias.width(wg_width)
    )


def _demo_coupler_symmetric(wg_width=0.5, gap=0.2):
    # c = coupler_symmetric(gap=gap, wg_width=wg_width)
    c = coupler_symmetric_biased(gap=gap, wg_width=wg_width)
    pp.write_gds(c)
    return c


if __name__ == "__main__":
    c = _demo_coupler_symmetric()
    pp.show(c)
