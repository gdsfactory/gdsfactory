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
