"""DBR gratings.

wavelength = 2*period*neff
period = wavelength/2/neff

dbr default parameters are from Stephen Lin thesis
<https://open.library.ubc.ca/cIRcle/collections/ubctheses/24/items/1.0388871>

Period: 318nm, width: 500nm, dw: 20 ~ 120 nm.
"""

from __future__ import annotations

__all__ = ["dbr", "dbr_cell"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions.filters.dbr import period, w1, w2
from gdsfactory.typings import CrossSectionSpec


@gf.cell_with_module_name(tags=["filters"])
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
        cross_section: cross_section spec.

    ```text
           l1      l2
        <-----><-------->
                _________
        _______|
    ```

    ```text
          w1       w2
        _______
               |_________
    ```
    """
    return cf.dbr_cell(w1=w1, w2=w2, l1=l1, l2=l2, cross_section=cross_section)


@gf.cell_with_module_name(tags=["filters"])
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

    ```text
           l1      l2
        <-----><-------->
                _________
        _______|
    ```

    ```text
          w1       w2       ...  n times
        _______
               |_________
    ```
    """
    return cf.dbr(
        w1=w1,
        w2=w2,
        l1=l1,
        l2=l2,
        n=n,
        cross_section=cross_section,
        straight_length=straight_length,
    )
