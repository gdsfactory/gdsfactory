"""based on phidl tutorial.

# Manipulating geometry 1 - Basic movement and rotation

There are several actions we can take to move and rotate the geometry.
These actions include movement, rotation, and reflection.
"""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component("demo")

    wg1 = c << gf.components.straight(length=10, width=1)
    wg2 = c << gf.components.straight(length=10, width=2, layer=(3, 0))

    # You can uncomment each of the following move commands
    # wg2.move([10, 1])  # Shift the second straight we created over by dx = 10, dy = 4
    # wg2.rotate(45)  # Rotate straight by 45 degrees around (0,0)
    # wg2.rotate(45, center=[5, 0])  # Rotate straight by 45 degrees around (5, 0)
    # wg2.mirror(p1=[1, 1], p2=[1, 3])  # Reflects wg across the line formed by p1 and p2

    c.show(show_ports=True)
