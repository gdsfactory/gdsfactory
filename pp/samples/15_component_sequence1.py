"""
This is a convenience function for cascading components. Usecase, is composite
waveguides such as phase modulators, where we need to keep track of multiple tapers,
doped sections, undopped, heaters etc...

The idea is to associate one symbol per type of section.
A section is uniquely defined by the component, its selected input and its selected output.

The mapping between symbols and components is supplied by a dictionnary.
The actual chain of components is supplied by a string or a list
"""

import pp
from pp.components import bend_circular
from pp.components.component_sequence import component_sequence
from pp.components.waveguide import waveguide
from pp.components.waveguide_heater import waveguide_heater


@pp.cell
def test_cutback_heater():
    # Define subcomponents
    bend_radius = 10.0
    bend180 = bend_circular(radius=bend_radius, start_angle=-90, theta=180)
    wg = waveguide(length=5.0)
    wg_heater = waveguide_heater(length=20.0)

    # Define a map between symbols and (component, input port, output port)
    string_to_device_in_out_ports = {
        "A": (bend180, "W0", "W1"),
        "B": (bend180, "W1", "W0"),
        "H": (wg_heater, "W0", "E0"),
        "-": (wg, "W0", "E0"),
    }

    # Generate a sequence
    # This is simply a chain of characters. Each of them represents a component
    # with a given input and and a given output

    sequence = "AB-H-H-H-H-BA"
    component = component_sequence(sequence, string_to_device_in_out_ports)
    assert component
    return component


if __name__ == "__main__":
    c = test_cutback_heater()
    pp.show(c)
