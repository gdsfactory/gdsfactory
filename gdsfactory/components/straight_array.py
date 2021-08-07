import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.port import auto_rename_ports
from gdsfactory.types import ComponentOrFactory


@gf.cell
def straight_array(
    n: int = 4,
    spacing: float = 4.0,
    straigth: ComponentOrFactory = straight_function,
    **kwargs
) -> Component:
    """Array of straights connected with grating couplers.

    useful to align the 4 corners of the chip

    Args:
        n: number of straights
        spacing: edge to edge straight spacing
        straigth: straigth straight Component or library
        **kwargs
    """

    c = Component()
    wg = straigth(**kwargs) if callable(straigth) else straigth

    for i in range(n):
        wref = c.add_ref(wg)
        wref.y += i * (spacing + wg.width)
        c.ports["E" + str(i)] = wref.ports["E0"]
        c.ports["W" + str(i)] = wref.ports["W0"]
    auto_rename_ports(c)
    return c


if __name__ == "__main__":
    c = straight_array(waveguide="nitride")
    c.show()
