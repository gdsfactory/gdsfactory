from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def spiral_inductor(
    width: float = 3.0,
    pitch: float = 3.0,
    turns: int = 16,
    outer_diameter: float = 800,
    tail: float = 50.0,
) -> Component:
    """Generates a spiral inductor to make superconducting resonator for qubit readout.

    See J. M. Hornibrook, J. I. Colless, A. C. Mahoney, X. G. Croot, S. Blanvillain, H. Lu, A. C. Gossard, D. J. Reilly;
    Frequency multiplexing for readout of spin qubits. Appl. Phys. Lett. 10 March 2014; 104 (10): 103108. https://doi.org/10.1063/1.4868107

    Args:
        width: width of the inductor track.
        pitch: distance between the inductor tracks.
        turns: number of full spriral turns.
        outer_diameter: size of the inductor.
        tail: length of the inner and outer tail.
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
