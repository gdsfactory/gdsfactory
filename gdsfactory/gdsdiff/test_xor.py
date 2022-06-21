import gdsfactory as gf
from gdsfactory.gdsdiff.gdsdiff import gdsdiff


def test_differences() -> None:
    straight = gf.partial(
        gf.components.straight,
        with_bbox=True,
        cladding_layers=None,
        add_pins=None,
        add_bbox=None,
    )
    c1 = straight(length=2)
    c2 = straight(length=3)
    c = gdsdiff(c1, c2)
    assert c.references[-1].area() == 0.5


def test_no_differences() -> None:
    straight = gf.partial(
        gf.components.straight,
        with_bbox=True,
        cladding_layers=None,
        add_pins=None,
        add_bbox=None,
    )
    c1 = straight(length=2)
    c2 = straight(length=2)
    c = gdsdiff(c1, c2)
    assert c.references[-1].area() == 0


if __name__ == "__main__":
    # test_no_differences()
    # test_differences()
    c1 = gf.components.straight(length=2)
    c2 = gf.components.straight(length=2)
    c = gdsdiff(c1, c2)
    c.show(show_ports=True)
