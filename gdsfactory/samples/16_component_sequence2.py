import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.component_sequence import component_sequence
from gdsfactory.components.straight import straight
from gdsfactory.components.straight_pin import straight_pin
from gdsfactory.components.taper import taper_strip_to_ridge


@gf.cell
def test_cutback_phase(
    straight_length: float = 100.0, bend_radius: float = 10.0, n: int = 2
) -> Component:
    """Modulator sections connected by bends"""
    # Define sub components
    bend180 = gf.components.bend_circular180(radius=bend_radius)
    pm_wg = straight_pin(length=straight_length)
    wg_short = straight(length=1.0)
    wg_short2 = straight(length=2.0)
    wg_heater = straight_pin(length=10.0)
    taper = taper_strip_to_ridge()

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "I": (taper, 1, 2),
        "O": (taper, 2, 1),
        "S": (wg_short, 1, 2),
        "P": (pm_wg, 1, 2),
        "A": (bend180, 1, 2),
        "B": (bend180, 2, 1),
        "H": (wg_heater, 1, 2),
        "-": (wg_short2, 1, 2),
    }

    # Generate a sequence
    # This is simply a chain of characters. Each of them represents a component
    # with a given input and and a given output

    repeated_sequence = "SIPOSASIPOSB"
    heater_seq = "-H-H-H-H-"
    sequence = repeated_sequence * n + "SIPO" + heater_seq
    component = component_sequence(
        sequence=sequence, symbol_to_component=symbol_to_component
    )
    return component


if __name__ == "__main__":
    c = test_cutback_phase(n=1)
    c.show()
