import gdsfactory as gf
from gdsfactory.gpdk import LAYER


def test_get_polygons() -> None:
    c = gf.c.rectangle(size=(10, 10), centered=True)

    p = c.get_polygons(layers=("WG",), by="tuple")
    key = next(iter(p.keys()))

    assert key == (1, 0)

    p = c.get_polygons(layers=("WG",), by="index")
    key = next(iter(p.keys()))
    assert key == LAYER.WG

    p = c.get_polygons(layers=("WG",), by="name")
    key = next(iter(p.keys()))
    assert key == "WG"


def test_get_polygons_instance_array() -> None:
    c = gf.Component()
    ref = c.add_ref(
        gf.c.rectangle(size=(1, 1)),
        columns=3,
        rows=2,
        column_pitch=10,
        row_pitch=20,
    )
    ref.move((100, 0))

    polygons = gf.functions.get_polygons(ref, layers=("WG",), by="name")["WG"]
    assert len(polygons) == 6
    assert sorted((p.bbox().left, p.bbox().bottom) for p in polygons) == sorted(
        (c.kcl.to_dbu(100 + 10 * i), c.kcl.to_dbu(20 * j))
        for i in range(3)
        for j in range(2)
    )


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
    assert c1.area(layer=layer) == c2.area(layer=layer), (
        f"{c1.area(layer=layer)} != {c2.area(layer=layer)}"
    )

    assert len(c2.ports) == len(c1.ports), f"{len(c2.ports)} != {len(c1.ports)}"


def test_area() -> None:
    c = gf.Component()
    _ = c << gf.c.rectangle(size=(10, 10), layer=(1, 0), centered=True)
    _ = c << gf.c.rectangle(size=(10, 10), layer=(1, 0), centered=True)
    area = c.area(layer=(1, 0))
    assert area == 100.0, f"{area} != 100"


def test_extract() -> None:
    c = gf.Component()
    r1 = c << gf.c.compass(size=(10, 10), layer=(1, 0))
    r2 = c << gf.c.compass(size=(10, 10), layer=(2, 0))
    r2.xmin = r1.xmax

    c1 = c.extract(layers=[(1, 0)])
    c2 = c.extract(layers=["WG"])
    area1 = c1.area(layer=(1, 0))
    area2 = c2.area(layer=(1, 0))

    area3 = c1.area(layer=(2, 0))
    assert area1 == 100.0, f"{area1} != 100"
    assert area2 == 100.0, f"{area2} != 100"
    assert area3 == 0.0, f"{area3} != 0"
