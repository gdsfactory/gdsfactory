from pp.components.coupler_ring import coupler_ring
from pp.components.waveguide import waveguide
from pp.drc import assert_on_2nm_grid
from pp.component import Component
from pp.config import call_if_func
from pp.name import autoname


@autoname
def ring_double(
    wg_width=0.5,
    gap=0.2,
    length_x=4,
    bend_radius=5,
    length_y=2,
    coupler=coupler_ring,
    waveguide=waveguide,
):
    """ double bus ring made of two couplers (ct: top, cb: bottom)
    connected with two vertical waveguides (wyl: left, wyr: right)

    .. code::

         --==ct==--
          |      |
          wl     wr length_y
          |      |
         --==cb==-- gap

          length_x
    """
    assert_on_2nm_grid(gap)

    coupler = call_if_func(
        coupler, gap=gap, wg_width=wg_width, bend_radius=bend_radius, length_x=length_x
    )
    waveguide = call_if_func(waveguide, width=wg_width, length=length_y)

    c = Component()
    cb = c << coupler
    ct = c << coupler
    wl = c << waveguide
    wr = c << waveguide

    wl.connect(port="E0", destination=cb.ports["N0"])
    ct.connect(port="N1", destination=wl.ports["W0"])
    wr.connect(port="W0", destination=ct.ports["N0"])
    c.add_port("E0", port=cb.ports["E0"])
    c.add_port("W0", port=cb.ports["W0"])
    c.add_port("E1", port=ct.ports["W0"])
    c.add_port("W1", port=ct.ports["E0"])
    return c


if __name__ == "__main__":
    import pp

    c = ring_double(with_pins=True)
    pp.show(c)
