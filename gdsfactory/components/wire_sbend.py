"""
"""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.wire import wire_corner, wire_straight


@gf.cell
def wire_sbend(dx: float = 20.0, dy: float = 10.0, **kwargs) -> Component:
    """Sbend corner

    Args:
        dx: length
        dy: height
        **kwargs: waveguide_settings
    """
    sx = wire_straight(length=dx / 2, **kwargs)
    sy = wire_straight(length=dy, **kwargs)
    bc = wire_corner(**kwargs)

    symbol_to_component = {
        "-": (sx, "DC_0", "DC_1"),
        "|": (sy, "DC_0", "DC_1"),
        "b": (bc, "DC_0", "DC_1"),
        "B": (bc, "DC_1", "DC_0"),
    }

    sequence = "-B|b-"
    c = gf.components.component_sequence(
        sequence=sequence, symbol_to_component=symbol_to_component
    )
    return c


if __name__ == "__main__":

    c = wire_sbend(width=5)
    c.show(show_ports=True)
