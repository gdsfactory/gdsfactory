import pp
from pp.component import Component
from pp.components.waveguide import waveguide as waveguide_function
from pp.port import auto_rename_ports
from pp.types import ComponentOrFactory


@pp.cell
def waveguide_array(
    n_waveguides: int = 4,
    spacing: float = 4.0,
    straigth: ComponentOrFactory = waveguide_function,
    **straigth_settings
) -> Component:
    """Array of waveguides connected with grating couplers.

    useful to align the 4 corners of the chip

    Args:
        n_waveguides: number of waveguides
        spacing: edge to edge waveguide spacing
        straigth: straigth waveguide Component or factory
        **straigth_settings
    """

    c = Component()
    wg = straigth(**straigth_settings) if callable(straigth) else straigth

    for i in range(n_waveguides):
        wref = c.add_ref(wg)
        wref.y += i * (spacing + wg.width)
        c.ports["E" + str(i)] = wref.ports["E0"]
        c.ports["W" + str(i)] = wref.ports["W0"]
    auto_rename_ports(c)
    return c


if __name__ == "__main__":
    c = waveguide_array()
    c.show()
