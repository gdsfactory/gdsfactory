"""when you add references you have to make sure they have unique names.

the cell decorator gives unique names to components that depend on their parameters
"""


from typing import Union

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def straight(width: Union[float, int] = 10, height: int = 1) -> Component:
    """Returns straight with automatic name."""
    wg = gf.Component("straight")
    wg.add_polygon([(0, 0), (width, 0), (width, height), (0, height)])
    wg.add_port(name="wgport1", midpoint=[0, height / 2], width=height, orientation=180)
    wg.add_port(
        name="wgport2", midpoint=[width, height / 2], width=height, orientation=0
    )
    return wg


def test_autoname() -> None:
    c = straight()
    assert c.name.split("_")[0] == "straight"


if __name__ == "__main__":
    c1 = straight()
    print(c1)

    c2 = straight(width=0.5)
    print(c2)

    c3 = straight(0.5)
    assert c2.name == c3.name
