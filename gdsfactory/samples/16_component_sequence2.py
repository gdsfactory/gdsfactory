from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.component_sequence import component_sequence
from gdsfactory.components.straight import straight
from gdsfactory.components.taper import taper_strip_to_ridge


@gf.cell
def cutback_phase(
    straight_length: float = 100.0, bend_radius: float = 12.0, n: int = 2
) -> Component:
    """Modulator sections connected by bends.

    Args:
        straight_length: length of the straight waveguides.
        bend_radius: radius of the bends.
        n: number of modulator sections.
    """
    # Define sub components
    bend180 = gf.components.bend_circular180(radius=bend_radius)
    pm_wg = gf.components.straight_pin(length=straight_length, taper=None)
    wg_short = straight(length=1.0)
    wg_short2 = straight(length=2.0)
    wg_heater = gf.components.straight_pin(length=10.0, taper=None)
    taper = taper_strip_to_ridge()

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "I": (taper, "o1", "o2"),
        "O": (taper, "o2", "o1"),
        "S": (wg_short, "o1", "o2"),
        "P": (pm_wg, "o1", "o2"),
        "A": (bend180, "o1", "o2"),
        "B": (bend180, "o2", "o1"),
        "H": (wg_heater, "o1", "o2"),
        "-": (wg_short2, "o1", "o2"),
    }

    # Generate a sequence
    # This is simply a chain of characters. Each of them represents a component
    # with a given input and and a given output
    repeated_sequence = "SIPOSASIPOSB"
    heater_seq = "-H-H-H-H-"
    sequence = repeated_sequence * n + "SIPO" + heater_seq
    return component_sequence(
        sequence=sequence, symbol_to_component=symbol_to_component
    )


def test_cutback_phase() -> None:
    assert cutback_phase()


if __name__ == "__main__":
    c = cutback_phase(n=1)
    c.show(show_ports=True)
