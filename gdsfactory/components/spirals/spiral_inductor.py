from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell_with_module_name
def spiral_inductor(
    width: float = 3.0,
    pitch: float = 3.0,
    turns: int = 16,
    outer_diameter: float = 800,
    tail: float = 50.0,
) -> Component:
    """Generates a spiral inductor for superconducting resonator applications, particularly in qubit readout circuits.

    This component creates a spiral inductor pattern commonly used in superconducting quantum circuits.
    The inductor is designed with a square spiral geometry, featuring inner and outer connection tails.

    See J. M. Hornibrook, J. I. Colless, A. C. Mahoney, X. G. Croot, S. Blanvillain, H. Lu, A. C. Gossard, D. J. Reilly;
    Frequency multiplexing for readout of spin qubits. Appl. Phys. Lett. 10 March 2014; 104 (10): 103108. https://doi.org/10.1063/1.4868107

    Args:
        width: Width of the inductor track in microns. Determines the cross-sectional area of the inductor.
        pitch: Distance between adjacent inductor tracks in microns. Affects the coupling between turns.
        turns: Number of complete spiral turns. Higher values increase inductance but require more space.
        outer_diameter: Overall size of the inductor in microns. Defines the maximum extent of the spiral.
        tail: Length of the inner and outer connection tails in microns. Used for connecting to other circuit elements.

    Returns:
        Component: A GDSFactory component containing the spiral inductor pattern.

    Example:
        ```python
        import gdsfactory as gf

        # Create a standard spiral inductor
        inductor = gf.components.spiral_inductor()

        # Create a custom spiral inductor with specific parameters
        custom_inductor = gf.components.spiral_inductor(
            width=2.0,
            pitch=2.5,
            turns=12,
            outer_diameter=600,
            tail=40.0
        )
        ```
    """
    # create the outer tail
    P = gf.path.straight(length=tail)
    P.end_angle -= 90
    for i in range(turns * 2):
        P += gf.path.arc(radius=outer_diameter / 2 - (pitch + width) * i / 2, angle=180)

    # create the inner tail
    P.end_angle += 90  # "Turn" 90 deg (left)
    P += gf.path.straight(length=tail)
    return gf.path.extrude(P, layer=(1, 0), width=width)


if __name__ == "__main__":
    c = spiral_inductor()
    c.show()
