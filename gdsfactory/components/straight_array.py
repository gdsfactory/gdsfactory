import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.straight import straight as straight_function
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
        wref.y += i * (spacing + wg.info.width)
        c.add_ports(wref.ports, prefix=str(i))

    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = straight_array()
    c.pprint_ports()
    c.show(show_ports=True)
