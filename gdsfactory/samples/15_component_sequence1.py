"""You can use component_sequence as a convenient function for cascading components, where you need to keep track of multiple tapers, doped sections, heaters etc...

The idea is to associate one symbol per type of section.
A section is uniquely defined by the component, input port name and output port name.

The mapping between symbols and components is supplied by a dictionary.
The actual chain of components is supplied by a string or a list

"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components import bend_circular
from gdsfactory.components.component_sequence import component_sequence
from gdsfactory.components.straight import straight
from gdsfactory.components.straight_pin import straight_pn


@gf.cell
def test_cutback_pn() -> Component:
    # Define subcomponents
    bend_radius = 10.0
    bend180 = bend_circular(radius=bend_radius, angle=180)
    wg = straight(length=5.0)
    wg_heater = straight_pn(length=50.0)

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (bend180, "o1", "o2"),
        "B": (bend180, "o2", "o1"),
        "H": (wg_heater, "o1", "o2"),
        "-": (wg, "o1", "o2"),
    }

    # Generate a sequence
    # This is simply a chain of characters. Each of them represents a component
    # with a given input and a given output

    sequence = "AB-H-H-H-H-BA"
    return component_sequence(
        sequence=sequence, symbol_to_component=symbol_to_component
    )


if __name__ == "__main__":
    c = test_cutback_pn()
    c.show(show_ports=True)
