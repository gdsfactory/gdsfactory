from typing import Callable

from pp.cell import cell
from pp.component import Component
from pp.components.bend_circular import bend_circular
from pp.components.coupler_ring import coupler_ring
from pp.components.waveguide import waveguide as waveguide_function
from pp.config import call_if_func
from pp.drc import assert_on_2nm_grid


@cell
def ring_single(
    wg_width: float = 0.5,
    gap: float = 0.2,
    bend_radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.001,
    coupler: Callable = coupler_ring,
    waveguide: Callable = waveguide_function,
    bend: Callable = bend_circular,
    pins: bool = False,
) -> Component:
    """Single bus ring made of a ring coupler (cb: bottom)
    connected with two vertical waveguides (wl: left, wr: right)
    two bends (bl, br) and horizontal waveguide (wg: top)

    Args:
        wg_width: waveguide width
        gap: gap between for coupler
        bend_radius: for the bend and coupler
        length_x: ring coupler length
        length_y: vertical waveguide length
        coupler: ring coupler function
        waveguide: waveguide function
        bend: bend function
        pins: add pins


    .. code::

          bl-wt-br
          |      |
          wl     wr length_y
          |      |
         --==cb==-- gap

          length_x

    .. plot::
      :include-source:

      import pp

      c = pp.c.ring_single(wg_width=0.5, gap=0.2, length_x=4, length_y=0.1, bend_radius=5)
      pp.plotgds(c)

    """
    bend_radius = float(bend_radius)
    assert_on_2nm_grid(gap)

    coupler = call_if_func(
        coupler, gap=gap, wg_width=wg_width, bend_radius=bend_radius, length_x=length_x
    )
    waveguide_side = call_if_func(waveguide, width=wg_width, length=length_y)
    waveguide_top = call_if_func(waveguide, width=wg_width, length=length_x)
    bend_ref = bend(width=wg_width, radius=bend_radius) if callable(bend) else bend

    c = Component()
    cb = c << coupler
    wl = c << waveguide_side
    wr = c << waveguide_side
    bl = c << bend_ref
    br = c << bend_ref
    wt = c << waveguide_top

    wl.connect(port="E0", destination=cb.ports["N0"])
    bl.connect(port="N0", destination=wl.ports["W0"])

    wt.connect(port="W0", destination=bl.ports["W0"])
    br.connect(port="N0", destination=wt.ports["E0"])
    wr.connect(port="W0", destination=br.ports["W0"])
    wr.connect(port="E0", destination=cb.ports["N1"])  # just for netlist

    c.add_port("E0", port=cb.ports["E0"])
    c.add_port("W0", port=cb.ports["W0"])
    if pins:
        pp.add_pins_to_references(c)
    return c


if __name__ == "__main__":
    import pp

    c = ring_single()
    cc = pp.add_pins(c)
    # print(c.settings)
    # print(c.get_settings())
    pp.show(cc)
