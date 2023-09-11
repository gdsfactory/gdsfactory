from __future__ import annotations

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component


@cell
def logo(text: str = "GDSFACTORY") -> Component:
    """Returns GDSfactory logo."""
    c = Component()
    elements = []
    for i, letter in enumerate(text):
        _ = c << gf.components.text(letter, layer=(i + 1, 0), size=10)
        elements.append(c)

    c.distribute(
        elements="all",  # either 'all' or a list of objects
        direction="x",  # 'x' or 'y'
        spacing=1,
        separation=True,
    )
    return c


if __name__ == "__main__":
    c = logo()
    c.show(show_ports=True)
