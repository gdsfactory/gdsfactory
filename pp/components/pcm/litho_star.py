from pp.name import autoname
from pp.components import rectangle

import pp


@autoname
def litho_star(num_lines=20, line_width=2, diameter=200, layer=0):
    """ Creates a circular-star shape from lines, used as a lithographic
    resolution test pattern

    .. plot::
      :include-source:

      import pp

      c = pp.c.litho_star()
      pp.plotgds(c)
    """
    D = pp.Component()

    degree = 180 / num_lines
    R1 = rectangle(size=(line_width, diameter), layer=layer)
    for i in range(num_lines):
        r1 = D.add_ref(R1).rotate(degree * i)
        r1.center = (0, 0)

    return D


if __name__ == "__main__":
    c = litho_star()
    pp.show(c)
