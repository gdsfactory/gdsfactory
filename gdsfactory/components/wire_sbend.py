import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.wire import wire_corner, wire_straight


@gf.cell
def wire_sbend(dx: float = 20.0, dy: float = 10.0, **kwargs) -> Component:
    """Sbend corner with manhattan wires

    Args:
        dx: length
        dy: height
        **kwargs: cross_section settings
    """
    sx = wire_straight(length=dx / 2, **kwargs)
    sy = wire_straight(length=dy, **kwargs)
    bc = wire_corner(**kwargs)

    symbol_to_component = {
        "-": (sx, "e1", "e2"),
        "|": (sy, "e1", "e2"),
        "b": (bc, "e2", "e1"),
        "B": (bc, "e1", "e2"),
    }

    sequence = "-B|b-"
    c = gf.components.component_sequence(
        sequence=sequence, symbol_to_component=symbol_to_component
    )
    c.auto_rename_ports()
    return c


if __name__ == "__main__":

    c = wire_sbend(width=5)
    c.show(show_ports=True)
    c.pprint_ports
