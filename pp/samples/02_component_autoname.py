"""One problem is that when we add references we have to make sure they have unique names.

The photonics package `pp` has a cell decorator that names the objects that it produces depending on the parameters that we pass them
"""


from typing import Union

import pp
from pp.component import Component


@pp.cell
def straight_cell(width: Union[float, int] = 10, height: int = 1) -> Component:
    """Returns straight with automatic name."""
    wg = pp.Component("straight")
    wg.add_polygon([(0, 0), (width, 0), (width, height), (0, height)])
    wg.add_port(name="wgport1", midpoint=[0, height / 2], width=height, orientation=180)
    wg.add_port(
        name="wgport2", midpoint=[width, height / 2], width=height, orientation=0
    )
    return wg


def test_autoname() -> None:
    c = straight_cell()
    assert c.name == "straight_cell"

    c = straight_cell(width=0.5)
    assert c.name == "straight_cell_W500n"


if __name__ == "__main__":
    c = straight_cell()
    print(c)

    c = straight_cell(width=0.5)
    print(c)
