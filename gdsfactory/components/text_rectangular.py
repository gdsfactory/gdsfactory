from typing import Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.manhattan_font import manhattan_text
from gdsfactory.types import Coordinate, Layer


@gf.cell
def text_rectangular(
    text: str = "abcd",
    size: float = 10.0,
    position: Coordinate = (0, 0),
    justify: str = "left",
    layers: Tuple[Layer, ...] = ((1, 0),),
) -> Component:
    """Returns Pixel based rectangular text.

    Args:
        text: text
        size: pixel size
        position: coordinate
        justify: left, right, center
        layers: layers
    """
    c = gf.Component()
    for layer in layers:
        c.add_ref(
            manhattan_text(
                text=text, size=size, position=position, justify=justify, layer=layer
            )
        )
    return c


if __name__ == "__main__":
    c = text_rectangular(
        text=".[,ABCDEFGHIKKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:/",
        size=4.0,
        justify="right",
        position=(120.5, 3),
        layers=[(1, 0), (2, 0)],
    )
    c.show()
