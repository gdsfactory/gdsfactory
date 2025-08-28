"""When you create components you have to make sure they have unique names.

the cell decorator gives unique names to components that depend on their parameters.
"""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c1 = gf.components.straight(length=5)
    print(c1.name)
    c2 = gf.components.straight(length=9)
    print(c2.name)
