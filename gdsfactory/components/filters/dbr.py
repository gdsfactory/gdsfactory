"""DBR gratings.

wavelength = 2*period*neff
period = wavelength/2/neff

dbr default parameters are from Stephen Lin thesis
https://open.library.ubc.ca/cIRcle/collections/ubctheses/24/items/1.0388871

Period: 318nm, width: 500nm, dw: 20 ~ 120 nm.
"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import CrossSectionSpec

period = 318e-3
w0 = 0.5
dw = 100e-3
w1 = w0 - dw / 2
w2 = w0 + dw / 2


@gf.cell_with_module_name
def dbr_cell(
    w1: float = w1,
    w2: float = w2,
    l1: float = period / 2,
    l2: float = period / 2,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Distributed Bragg Reflector unit cell.

    Args:
        w1: thin width in um.
        l1: thin length in um.
        w2: thick width in um.
        l2: thick length in um.
        n: number of periods.
        cross_section: cross_section spec.

    .. code::

           l1      l2
        <-----><-------->
                _________
        _______|

          w1       w2
        _______
               |_________
    """
    l1 = snap_to_grid(l1)
    l2 = snap_to_grid(l2)
    w1 = snap_to_grid(w1, 2)
    w2 = snap_to_grid(w2, 2)
    xs1 = gf.get_cross_section(cross_section, width=w1)
    xs2 = gf.get_cross_section(cross_section, width=w2)

    c = Component()
    c1 = c << gf.c.straight(length=l1, cross_section=xs1)
    c2 = c << gf.c.straight(length=l2, cross_section=xs2)
    c2.connect(port="o1", other=c1.ports["o2"], allow_width_mismatch=True)
    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c2.ports["o2"])
    c.flatten()
    return c


@gf.cell_with_module_name
def dbr(
    w1: float = w1,
    w2: float = w2,
    l1: float = period / 2,
    l2: float = period / 2,
    n: int = 10,
    cross_section: CrossSectionSpec = "strip",
    straight_length: float = 10e-3,
) -> Component:
    """Distributed Bragg Reflector.

    Args:
        w1: thin width in um.
        w2: thick width in um.
        l1: thin length in um.
        l2: thick length in um.
        n: number of periods.
        cross_section: cross_section spec.
        straight_length: length of the straight section between cutbacks.

    .. code::

           l1      l2
        <-----><-------->
                _________
        _______|

          w1       w2       ...  n times
        _______
               |_________
    """
    c = Component()
    xs = gf.get_cross_section(cross_section)
    s1 = c << gf.c.straight(cross_section=xs, length=straight_length)
    s2 = c << gf.c.straight(cross_section=xs, length=straight_length)

    cell = dbr_cell(w1=w1, w2=w2, l1=l1, l2=l2, cross_section=cross_section)
    ref = c.add_ref(cell, columns=n, rows=1, column_pitch=l1 + l2)

    s1.connect(port="o1", other=cell.ports["o1"], allow_width_mismatch=True)
    s2.connect(port="o1", other=cell.ports["o2"], allow_width_mismatch=True)
    s2.xmin = ref.xmax

    c.add_port("o1", port=s1.ports["o2"])
    return c


if __name__ == "__main__":
    # c = dbr(w1=0.5, w2=0.6, l1=0.2, l2=0.3, n=10)
    c = dbr(n=2)
    # c = dbr_cell()
    # c.assert_ports_on_grid()
    c.show()
