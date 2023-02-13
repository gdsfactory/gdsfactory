from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.component_sequence import component_sequence
from gdsfactory.components.mmi2x2 import mmi2x2
from gdsfactory.typings import ComponentSpec


@gf.cell
def cutback_2x2(
    component: ComponentSpec = mmi2x2,
    cols: int = 4,
    port1: str = "o1",
    port2: str = "o2",
    port3: str = "o3",
    port4: str = "o4",
) -> Component:
    """Returns a daisy chain of 2x2 couplers for measuring their loss.

    Args:
        component: for cutback.
        cols: number of columns/components.
        port1: name of first optical port.
        port2: name of second optical port.
        port3: name of third optical port.
        port4: name of fourth optical port.
    """
    component = gf.get_component(component)

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (component, port1, port3),
        "B": (component, port4, port2),
    }

    # Generate the sequence of staircases

    s = "AB" * cols

    seq = component_sequence(sequence=s, symbol_to_component=symbol_to_component)

    c = gf.Component()
    ref = c << seq
    c.add_ports(ref.ports)

    n = len(s) - 2
    c.copy_child_info(component)
    c.info["components"] = n
    return c


if __name__ == "__main__":
    c = cutback_2x2(component=gf.c.coupler, cols=2)
    c.show(show_ports=True)
