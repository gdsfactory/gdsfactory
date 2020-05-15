import pp
from pp.components.bend_circular import bend_circular
from pp.components.coupler_ring import coupler_ring
from pp.components.waveguide import waveguide
from pp.drc import assert_on_2nm_grid


@pp.autoname
def ring_single(
    wg_width=0.5,
    gap=0.2,
    length_x=4,
    bend_radius=5,
    length_y=2,
    coupler=coupler_ring,
    waveguide=waveguide,
    bend=bend_circular,
):
    """ single bus ring made of two couplers (ct: top, cb: bottom)
    connected with two vertical waveguides (wyl: left, wyr: right)

    .. code::

          bl-wt-br
          |      |
          wl     wr length_y
          |      |
         --==cb==-- gap

          length_x
    """
    assert_on_2nm_grid(gap)

    coupler = pp.call_if_func(
        coupler, gap=gap, wg_width=wg_width, bend_radius=bend_radius, length_x=length_x
    )
    waveguide_side = pp.call_if_func(waveguide, width=wg_width, length=length_y)
    waveguide_top = pp.call_if_func(waveguide, width=wg_width, length=length_x)
    bend = pp.call_if_func(bend, width=wg_width, radius=bend_radius)

    c = pp.Component()
    cb = c << coupler
    wl = c << waveguide_side
    wr = c << waveguide_side
    bl = c << bend
    br = c << bend
    wt = c << waveguide_top

    wl.connect(port="E0", destination=cb.ports["N0"])
    bl.connect(port="N0", destination=wl.ports["W0"])

    wt.connect(port="W0", destination=bl.ports["W0"])
    br.connect(port="N0", destination=wt.ports["E0"])
    wr.connect(port="W0", destination=br.ports["W0"])
    return c


if __name__ == "__main__":
    c = ring_single()
    pp.show(c)
