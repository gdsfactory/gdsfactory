from typing import Callable, Iterable, Tuple

import pp
from pp.component import Component
from pp.components.bend_s import bend_s


@pp.cell
def coupler_symmetric(
    bend: Callable = bend_s,
    gap: float = 0.234,
    wg_width: float = 0.5,
    layer: Tuple[int, int] = pp.LAYER.WG,
    layers_cladding: Iterable[Tuple[int, int]] = (pp.LAYER.WGCLAD,),
    cladding_offset: float = 3.0,
    dy: float = 5.0,
) -> Component:
    r"""Two coupled waveguides with bends.

    Args:
        bend: bend or factory
        gap:
        wg_width:
        layer
        layers_cladding
        cladding_offset
        dy: port to port vertical spacing

    .. plot::
      :include-source:

      import pp

      c = pp.c.coupler_symmetric()
      pp.plotgds(c)

    .. code::

            _ E1
           /     |
          /      |
         = gap   | dy
          \      |
           \_    |
              E0

    """
    bend_component = (
        bend(
            width=wg_width,
            layer=layer,
            layers_cladding=layers_cladding,
            cladding_offset=cladding_offset,
            height=(dy - gap - wg_width) / 2,
        )
        if callable(bend)
        else bend
    )

    w = bend_component.ports["W0"].width
    y = (w + gap) / 2

    c = pp.Component()
    top_bend = bend_component.ref(position=(0, y), port_id="W0")
    bottom_bend = bend_component.ref(position=(0, -y), port_id="W0", v_mirror=True)

    c.add(top_bend)
    c.add(bottom_bend)

    c.absorb(top_bend)
    c.absorb(bottom_bend)

    port_width = 2 * w + gap
    c.add_port(name="W0", midpoint=[0, 0], width=port_width, orientation=180)
    c.add_port(port=bottom_bend.ports["E0"], name="E0")
    c.add_port(port=top_bend.ports["E0"], name="E1")
    c.length = bend_component.length
    c.min_bend_radius = bend_component.min_bend_radius
    return c


@pp.cell
def coupler_symmetric_biased(bend=bend_s, gap=0.2, wg_width=0.5):
    return coupler_symmetric(
        bend=bend, gap=pp.bias.gap(gap), wg_width=pp.bias.width(wg_width)
    )


if __name__ == "__main__":
    c = coupler_symmetric_biased(gap=0.2, wg_width=0.5)
    # c.show()
    # c.pprint()

    for dy in [2, 3, 4, 5]:
        c = coupler_symmetric(gap=0.2, wg_width=0.5, dy=dy)
        print(f"dy={dy}, min_bend_radius = {c.min_bend_radius}")
