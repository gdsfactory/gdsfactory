import gdsfactory as gf
from gdsfactory.generic_tech import LAYER


def test_get_polygons():
    c = gf.c.rectangle(size=(10, 10), centered=True)

    p = c.get_polygons(layers=("WG",), by="tuple")
    key = list(p.keys())[0]

    assert key == (1, 0)

    p = c.get_polygons(layers=("WG",), by="index")
    key = list(p.keys())[0]
    assert key == LAYER.WG

    p = c.get_polygons(layers=("WG",), by="name")
    key = list(p.keys())[0]
    assert key == "WG"


def test_trim() -> None:
    layer = (1, 0)
    c1 = gf.c.rectangle(size=(11, 11), centered=True, layer=layer).dup()
    c2 = gf.functions.trim(
        c1,
        domain=((-5, -5), (-5, +5), (+5, +5), (+5, -5)),
    )
    area = c2.area(layer=layer)
    assert area == 100, f"{area} != 100"
    assert len(c2.ports) == len(c1.ports), f"{len(c2.ports)} != {len(c1.ports)}"


def test_trim_no_clipping() -> None:
    layer = (1, 0)
    c1 = gf.c.rectangle(size=(10, 10), centered=True, layer=layer).dup()
    c2 = gf.functions.trim(
        c1,
        domain=((-5, -5), (-5, +5), (+5, +5), (+5, -5)),
    )
    assert c1.area(layer=layer) == c2.area(
        layer=layer
    ), f"{c1.area(layer=layer)} != {c2.area(layer=layer)}"

    assert len(c2.ports) == len(c1.ports), f"{len(c2.ports)} != {len(c1.ports)}"


def test_area() -> None:
    c = gf.Component()
    _ = c << gf.c.rectangle(size=(10, 10), layer=(1, 0), centered=True)
    _ = c << gf.c.rectangle(size=(10, 10), layer=(1, 0), centered=True)
    area = c.area(layer=(1, 0))
    assert area == 100.0, f"{area} != 100"


if __name__ == "__main__":
    test_area()
