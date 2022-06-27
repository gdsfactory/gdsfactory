"""You can add pins in a pin layer to clearly see the component ports."""

import gdsfactory as gf
from gdsfactory.types import LayerSpec


@gf.cell
def straight(
    length: float = 5.0, width: float = 1.0, layer: LayerSpec = (2, 0)
) -> gf.Component:
    """Returns straight Component.

    Args:
        length: of the straight.
        width: in um.
        layer: layer spec
    """
    wg = gf.Component("straight_sample")
    wg.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)
    wg.add_port(
        name="o1", midpoint=(0, width / 2), width=width, orientation=180, layer=layer
    )
    wg.add_port(
        name="o2", midpoint=(length, width / 2), width=width, orientation=0, layer=layer
    )
    return wg


def test_straight(data_regression):
    component = straight()
    data_regression.check(component.to_dict())
    assert straight()


if __name__ == "__main__":
    wg = straight(decorator=gf.add_pins.add_pins)

    # By default show adds pins, so you don't need it to show_ports
    wg.show(show_ports=False)
