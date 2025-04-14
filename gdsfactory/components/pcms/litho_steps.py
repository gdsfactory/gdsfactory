from __future__ import annotations

import gdsfactory as gf
from gdsfactory import components as pc
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def litho_steps(
    line_widths: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0),
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
        text=f"{line_widths[-1]!s}", size=height, justify="center", layer=layer
    )

    ref = D.add_ref(T1)
    ref.rotate(90)
    ref.movex(-height / 10)

    R1 = pc.rectangle(size=(line_spacing, height), layer=layer)
    D.add_ref(R1).movey(-height)
    count = 0.0
    for i in reversed(line_widths):
        count += line_spacing + i
        R2 = pc.rectangle(size=(i, height), layer=layer)
        r = D.add_ref(R1)
        r.movex(count)
        r.movey(-height)
        r = D.add_ref(R2)
        r.movex(count - i)

    return D


if __name__ == "__main__":
    c = litho_steps()
    c.show()
