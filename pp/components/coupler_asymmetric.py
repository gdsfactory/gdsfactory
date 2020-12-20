from typing import Callable

import pp
from pp.component import Component
from pp.components.bend_s import bend_s
from pp.components.waveguide import waveguide as waveguide_function


@pp.cell
def coupler_asymmetric(
    bend: Callable = bend_s,
    waveguide: Callable = waveguide_function,
    gap: float = 0.234,
    wg_width: float = 0.5,
) -> Component:
    """ bend coupled to straight waveguide

    Args:
        bend:
        waveguide: waveguide factory
        gap: um
        wg_width

    .. plot::
      :include-source:

      import pp

      c = pp.c.coupler_asymmetric()
      pp.plotgds(c)

    """
    bend = bend(width=wg_width)
    wg = waveguide(width=wg_width)

    w = bend.ports["W0"].width
    y = (w + gap) / 2

    c = pp.Component()
    wg = wg.ref(position=(0, y), port_id="W0")
    bottom_bend = bend.ref(position=(0, -y), port_id="W0", v_mirror=True)

    c.add(wg)
    c.add(bottom_bend)

    # Using absorb here to have a flat cell and avoid
    # to have deeper hierarchy than needed
    c.absorb(wg)
    c.absorb(bottom_bend)

    port_width = 2 * w + gap
    c.add_port(name="W0", midpoint=[0, 0], width=port_width, orientation=180)
    c.add_port(port=bottom_bend.ports["E0"], name="E0")
    c.add_port(port=wg.ports["E0"], name="E1")

    return c


if __name__ == "__main__":
    c = coupler_asymmetric(wg_width=0.6, gap=0.4)
    pp.show(c)
