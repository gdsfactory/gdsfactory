from gdstk import Polygon

import gdsfactory as gf


def test_move() -> None:
    c = gf.Component()
    mzi = c.add_ref(gf.components.mzi())
    bend = c.add_ref(gf.components.bend_euler())
    bend.move("o1", mzi.ports["o2"])


def test_get_polygons() -> None:
    ref = gf.components.straight()
    p0 = ref.get_polygons(by_spec="WG", as_array=False)
    p1 = ref.get_polygons(by_spec=(1, 0), as_array=True)
    p2 = ref.get_polygons(by_spec=(1, 0), as_array=False)

    p3 = ref.get_polygons(by_spec=True, as_array=True)[(1, 0)]
    p4 = ref.get_polygons(by_spec=True, as_array=False)[(1, 0)]

    assert len(p1) == len(p2) == len(p3) == len(p4) == 1 == len(p0)
    assert p1[0].dtype == p3[0].dtype == float
    assert isinstance(p2[0], Polygon)
    assert isinstance(p4[0], Polygon)


def test_get_polygons_ref() -> None:
    ref = gf.components.straight().ref()
    p0 = ref.get_polygons(by_spec="WG", as_array=False)
    p1 = ref.get_polygons(by_spec=(1, 0), as_array=True)
    p2 = ref.get_polygons(by_spec=(1, 0), as_array=False)

    p3 = ref.get_polygons(by_spec=True, as_array=True)[(1, 0)]
    p4 = ref.get_polygons(by_spec=True, as_array=False)[(1, 0)]

    assert len(p1) == len(p2) == len(p3) == len(p4) == 1 == len(p0)
    assert p1[0].dtype == p3[0].dtype == float
    assert isinstance(p2[0], Polygon)
    assert isinstance(p4[0], Polygon)


def test_pads_no_orientation() -> None:
    c = gf.Component()
    pt = c << gf.components.pad()
    pb = c << gf.components.pad()
    pb.connect("pad", pt["pad"])


def test_rotation_of_ports_with_no_orientation():
    c = gf.Component()
    pt = c << gf.components.pad_array(orientation=None, columns=3)
    pb = c << gf.components.pad_array(orientation=None, columns=3)
    pt.move((70, 200))
    pt.rotate(90)
    route = gf.routing.get_route_electrical(
        pt.ports["e11"], pb.ports["e11"], bend="wire_corner"
    )
    c.add(route.references)
