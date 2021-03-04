import pp
from pp.component import Component
from pp.components import bend_circular
from pp.components.component_sequence import component_sequence
from pp.components.taper import taper_strip_to_ridge
from pp.components.waveguide import waveguide
from pp.components.waveguide_heater import waveguide_heater
from pp.components.waveguide_pin import waveguide_pin


@pp.cell
def test_cutback_phase(
    straight_length: float = 100.0, bend_radius: float = 10.0, n: int = 2
) -> Component:
    """ Modulator sections connected by bends """
    # Define sub components
    bend180 = bend_circular(radius=bend_radius, angle=180)
    pm_wg = waveguide_pin(length=straight_length)
    wg_short = waveguide(length=1.0)
    wg_short2 = waveguide(length=2.0)
    wg_heater = waveguide_heater(length=10.0)
    taper = taper_strip_to_ridge()

    # Define a map between symbols and (component, input port, output port)
    string_to_device_in_out_ports = {
        "I": (taper, "1", "wg_2"),
        "O": (taper, "wg_2", "1"),
        "S": (wg_short, "W0", "E0"),
        "P": (pm_wg, "W0", "E0"),
        "A": (bend180, "W0", "W1"),
        "B": (bend180, "W1", "W0"),
        "H": (wg_heater, "W0", "E0"),
        "-": (wg_short2, "W0", "E0"),
    }

    # Generate a sequence
    # This is simply a chain of characters. Each of them represents a component
    # with a given input and and a given output

    repeated_sequence = "SIPOSASIPOSB"
    heater_seq = "-H-H-H-H-"
    sequence = repeated_sequence * n + "SIPO" + heater_seq
    component = component_sequence(sequence, string_to_device_in_out_ports)

    assert component
    return component


if __name__ == "__main__":
    c = test_cutback_phase()
    c.show()
