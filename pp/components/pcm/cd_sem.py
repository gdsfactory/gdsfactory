""" CD SEM structures
"""

import pp

from pp.components.bend_circular import bend_circular
from pp.components.bend_circular import bend_circular_trenches

from pp.components.waveguide import waveguide_trenches
from pp.layers import LAYER
from pp.port import rename_ports_by_orientation


def square_middle(side=0.5, layer=LAYER.WG):
    component = pp.Component()
    a = side / 2
    component.add_polygon([(-a, -a), (a, -a), (a, a), (-a, a)], layer=layer)
    return component


def text(t="U"):
    return pp.c.text(text=t, layer=pp.LAYER.WG, size=5)


CENTER_SHAPES_MAP = {"S": square_middle, "U": text("U"), "D": text("L")}


@pp.autoname
def cdsem_straight(w, dw, spacing=5.0, length=20.0):
    """
    w
    dw
    """

    c = pp.Component()

    c.move(c.size_info.cc, (0, 0))
    return c


@pp.autoname
def cdsem_straight_density(
    wg_width=0.372, trench_width=0.304, x=500, y=50.0, margin=2.0
):
    """ horizontal grating etch lines

    TE: 676nm pitch, 304nm gap, 372nm line
    TM: 1110nm pitch, 506nm gap, 604nm line

    Args:
        w: wg_width
        s: trench_width
    """
    c = pp.Component()
    period = wg_width + trench_width
    n_o_lines = int((y - 2 * margin) / period)
    length = x - 2 * margin

    slab = pp.c.rectangle_centered(x=x, y=y, layer=LAYER.WG)
    slab_ref = c.add_ref(slab)
    c.absorb(slab_ref)

    tooth = pp.c.rectangle_centered(x=length, y=trench_width, layer=LAYER.SLAB150)

    for i in range(n_o_lines):
        _tooth = c.add_ref(tooth)
        _tooth.movey((-n_o_lines / 2 + 0.5 + i) * period)
        c.absorb(_tooth)

    c.move(c.size_info.cc, (0, 0))
    return c


@pp.autoname
def cdsem_target(width_center=0.5):
    radii = [5.0, 10.0]
    c = pp.Component()
    a = 1.0
    w = 3 * a / 4
    c.add_ref(square_middle())
    ctr = c.add_ref(square_middle())
    ctl = c.add_ref(square_middle())
    cbr = c.add_ref(square_middle())
    cbl = c.add_ref(square_middle())

    ctr.move((w, w))
    ctl.move((-w, w))
    cbr.move((w, -w))
    cbl.move((-w, -w))

    w_min = width_center * 0.92
    w0 = width_center
    w_max = width_center * 1.08

    for radius in radii:
        b = a + radius
        _b_tr = bend_circular(radius=radius, width=w0)
        b_tr = _b_tr.ref(position=(b, a), rotation=90, port_id="W0")

        _b_bl = bend_circular(radius=radius, width=w0)
        b_bl = _b_bl.ref(position=(-b, -a), rotation=270, port_id="W0")

        _b_br = bend_circular(radius=radius, width=w_max)
        b_br = _b_br.ref(position=(a, -b), rotation=0, port_id="W0")

        _b_tl = bend_circular(radius=radius, width=w_min)
        b_tl = _b_tl.ref(position=(-a, b), rotation=180, port_id="W0")

        c.add([b_tr, b_tl, b_bl, b_br])

    return c


@pp.autoname
def cdsem_uturn(
    width=0.5, cladding_offset=3.0, radius=10, symbol_bot="S", symbol_top="D"
):
    c = pp.Component()
    r = radius

    # bend90 = bend_circular_deep_rib(width=width, radius=r, cladding_offset=cladding_offset)
    # wg = wg_deep_rib(width=width, length=2 * r)

    bend90 = bend_circular_trenches(width=width, radius=r)
    wg = waveguide_trenches(width=width, length=2 * r)

    # bend90.ports()
    rename_ports_by_orientation(bend90)

    # Add the U-turn on waveguide layer
    b1 = c.add_ref(bend90)
    b2 = c.add_ref(bend90)

    b2.connect("N0", b1.ports["W0"])

    wg1 = c.add_ref(wg)
    wg1.connect("W0", b1.ports["N0"])

    wg2 = c.add_ref(wg)
    wg2.connect("W0", b2.ports["W0"])

    # Add symbols

    sym1 = c.add_ref(CENTER_SHAPES_MAP[symbol_bot]())
    sym1.movey(r)
    sym2 = c.add_ref(CENTER_SHAPES_MAP[symbol_top])
    sym2.movey(2 * r)

    c.move(c.size_info.cc, (0, 0))
    return c


if __name__ == "__main__":
    # c = cdsem_straight_density()
    c = cdsem_uturn()
    pp.show(c)

    # c = pcm_bend()
    # pp.write_gds(c)
    # pp.show(c)
