from __future__ import annotations

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@cell
def logo(text: str = "GDSFACTORY", layer: LayerSpec | None = None) -> Component:
    """Returns GDSfactory logo.

    Args:
        text: text to write.
        layer: optional layer to use for the text.
    """
    c = Component()
    elements = []
    for i, letter in enumerate(text):
        _ = c << gf.components.text(letter, layer=layer or (i + 1, 0), size=10)
        elements.append(c)

    c.distribute(
        elements="all",  # either 'all' or a list of objects
        direction="x",  # 'x' or 'y'
        spacing=1,
        separation=True,
    )
    return c


if __name__ == "__main__":
    c = logo(layer=(1, 0))
    c.show(show_ports=True)
