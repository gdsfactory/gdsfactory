"""When you create components you have to make sure they have unique names.

the cell decorator gives unique names to components that depend on their
parameters

"""

import gdsfactory as gf


def test_autoname() -> None:
    c1 = gf.components.straight(length=5)
    assert c1.name.split("_")[0] == "straight"
