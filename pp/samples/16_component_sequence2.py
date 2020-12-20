import pp
from pp import LAYER
from pp.components import bend_circular
from pp.components.component_sequence import component_sequence
from pp.components.taper import taper_strip_to_ridge
from pp.components.waveguide import _arbitrary_straight_waveguide, waveguide
from pp.components.waveguide_heater import waveguide_heater


@pp.cell
def phase_modulator_waveguide(length, wg_width=0.5, cladding=3.0, si_outer_clad=1.0):
    """
    Phase modulator waveguide mockup
    """
    a = wg_width / 2
    b = a + cladding
    c = b + si_outer_clad

    windows = [
        (-c, -b, LAYER.WG),
        (-b, -a, LAYER.SLAB90),
        (-a, a, LAYER.WG),
        (a, b, LAYER.SLAB90),
        (b, c, LAYER.WG),
    ]

    component = _arbitrary_straight_waveguide(length=length, windows=windows)
    return component


@pp.cell
def test_cutback_phase(straight_length=100.0, bend_radius=10.0, n=2):
    """ Modulator sections connected by bends """
    # Define sub components
    bend180 = bend_circular(radius=bend_radius, start_angle=-90, theta=180)
    pm_wg = phase_modulator_waveguide(length=straight_length)
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
    pp.show(c)
