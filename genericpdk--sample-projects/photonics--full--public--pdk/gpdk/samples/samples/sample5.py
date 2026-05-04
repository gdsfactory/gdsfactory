"""Pack."""

import gdsfactory as gf


@gf.cell
def sample5_path():
    p = gf.Path()
    p += gf.path.arc(radius=10, angle=90)  # Circular arc
    p += gf.path.straight(length=10)  # Straight section
    p += gf.path.euler(radius=3, angle=-90)  # Euler bend (aka "racetrack" curve)
    p += gf.path.straight(length=40)
    p += gf.path.arc(radius=8, angle=-45)
    p += gf.path.straight(length=10)
    p += gf.path.arc(radius=8, angle=45)
    p += gf.path.straight(length=10)
    return p.extrude(layer=(3, 0), width=1.5)
