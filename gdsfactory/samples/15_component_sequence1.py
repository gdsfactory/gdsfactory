"""
This is a convenience function for cascading components. Usecase, is composite
straights such as phase modulators, where we need to keep track of multiple tapers,
doped sections, undopped, heaters etc...

The idea is to associate one symbol per type of section.
A section is uniquely defined by the component, its selected input and its selected output.

The mapping between symbols and components is supplied by a dictionnary.
The actual chain of components is supplied by a string or a list
"""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components import bend_circular
from gdsfactory.components.component_sequence import component_sequence
from gdsfactory.components.straight import straight
from gdsfactory.components.straight_heater import straight_heater


@gf.cell
def test_cutback_heater() -> Component:
    # Define subcomponents
    bend_radius = 10.0
    bend180 = bend_circular(radius=bend_radius, angle=180)
    wg = straight(length=5.0)
    wg_heater = straight_heater(length=20.0)

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (bend180, "W0", "W1"),
        "B": (bend180, "W1", "W0"),
        "H": (wg_heater, "W0", "E0"),
        "-": (wg, "W0", "E0"),
    }

    # Generate a sequence
    # This is simply a chain of characters. Each of them represents a component
    # with a given input and and a given output

    sequence = "AB-H-H-H-H-BA"
    component = component_sequence(
        sequence=sequence, symbol_to_component=symbol_to_component
    )
    return component


if __name__ == "__main__":
    c = test_cutback_heater()
    c.show()
