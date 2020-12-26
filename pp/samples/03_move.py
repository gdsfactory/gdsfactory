""" based on phidl tutorial

# Manipulating geometry 1 - Basic movement and rotation

There are several actions we can take to move and rotate the geometry.  These
actions include movement, rotation, and reflection.

"""


import pp

if __name__ == "__main__":
    c = pp.Component()

    wg1 = c << pp.c.waveguide(length=10, width=1)
    wg2 = c << pp.c.waveguide(length=10, width=2, layer=pp.LAYER.SLAB90)

    # wg2.move([10, 1])  # Shift the second waveguide we created over by dx = 10, dy = 4

    # You can unconmment and play with the following move commands

    # wg2.rotate(45) # Rotate waveguide by 45 degrees around (0,0)
    # wg2.rotate(45, center=[5, 0])  # Rotate waveguide by 45 degrees around (5, 0)
    # wg2.reflect(p1=[1, 1], p2=[1, 3])  # Reflects wg across the line formed
    # by p1 and p2

    pp.show(c)
