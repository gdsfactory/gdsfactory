from pp.cell import cell
from pp.component import Component
from pp.components.rectangle import rectangle


@cell
def litho_star(
    num_lines: int = 20, line_width: int = 2, diameter: int = 200, layer: int = 0
) -> Component:
    """ Creates a circular-star shape from lines, used as a lithographic
    resolution test pattern

    .. plot::
      :include-source:

      import pp

      c = pp.c.litho_star()
      pp.plotgds(c)
    """
    D = Component()

    degree = 180 / num_lines
    R1 = rectangle(size=(line_width, diameter), layer=layer)
    for i in range(num_lines):
        r1 = D.add_ref(R1).rotate(degree * i)
        r1.center = (0, 0)

    return D


if __name__ == "__main__":
    import pp

    c = litho_star()
    pp.show(c)
