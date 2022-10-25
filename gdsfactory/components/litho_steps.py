from typing import Tuple

import gdsfactory as gf
from gdsfactory import components as pc
from gdsfactory.component import Component
from gdsfactory.types import LayerSpec


@gf.cell
def litho_steps(
    line_widths: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0),
    line_spacing: float = 10.0,
    height: float = 100.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Positive + negative tone linewidth test.

    used for lithography resolution test patterning
    based on phidl

    Args:
        line_widths: in um.
        line_spacing: in um.
        height: in um.
        layer: Specific layer to put the ruler geometry on.
    """
    D = gf.Component()

    height /= 2
    T1 = pc.text(
        text=f"{str(line_widths[-1])}", size=height, justify="center", layer=layer
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
    c.show(show_ports=True)
