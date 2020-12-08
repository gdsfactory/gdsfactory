from typing import Callable

import pp
from pp.components.waveguide import waveguide as waveguide_function
from pp.port import deco_rename_ports


@pp.cell
@deco_rename_ports
def waveguide_array(
    n_waveguides: int = 4,
    spacing: float = 4.0,
    waveguide: Callable = waveguide_function,
) -> pp.Component:
    """array of waveguides connected with grating couplers
    useful to align the 4 corners of the chip

    Args:
        n_waveguides: number of waveguides
        spacing: edge to edge waveguide spacing
        waveguide: waveguide Component or factory
    """

    c = pp.Component()
    w = waveguide()

    for i in range(n_waveguides):
        wref = c.add_ref(w)
        wref.y += i * (spacing + w.width)
        c.ports["E" + str(i)] = wref.ports["E0"]
        c.ports["W" + str(i)] = wref.ports["W0"]
    return c


if __name__ == "__main__":
    c = waveguide_array()
    pp.show(c)
