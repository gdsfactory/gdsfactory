from typing import List, Tuple

import pp
from pp import components as pc
from pp.cell import cell
from pp.component import Component


@cell
def litho_steps(
    line_widths: List[float] = (1.0, 2.0, 4.0, 8.0, 16.0),
    line_spacing: float = 10.0,
    height: float = 100.0,
    layer: Tuple[int, int] = pp.LAYER.WG,
) -> Component:
    """ Produces a positive + negative tone linewidth test, used for
    lithography resolution test patterning

    Args:
        line_widths:
        line_spacing:
        height:
        layer:

    .. plot::
      :include-source:

      import pp

      c = pp.c.litho_steps()
      pp.plotgds(c)

    """
    D = pp.Component()

    height = height / 2
    T1 = pc.text(
        text="%s" % str(line_widths[-1]), size=height, justify="center", layer=layer
    )
    D.add_ref(T1).rotate(90).movex(-height / 10)
    R1 = pc.rectangle(size=(line_spacing, height), layer=layer)
    D.add_ref(R1).movey(-height)
    count = 0
    for i in reversed(line_widths):
        count += line_spacing + i
        R2 = pc.rectangle(size=(i, height), layer=layer)
        D.add_ref(R1).movex(count).movey(-height)
        D.add_ref(R2).movex(count - i)

    return D


if __name__ == "__main__":
    c = litho_steps()
    pp.show(c)
