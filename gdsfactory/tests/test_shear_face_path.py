import numpy as np
import pytest
import shapely.geometry as sg

import gdsfactory as gf

DEMO_PORT_ANGLE = 10


def assert_polygon_equals(coords_expected, coords_actual):
    coords_expected = gf.snap.snap_to_grid(coords_expected)
    coords_actual = gf.snap.snap_to_grid(coords_actual)
    shape_expected = sg.polygon.orient(sg.Polygon(coords_expected))
    shape_actual = sg.polygon.orient(sg.Polygon(coords_actual))

    assert shape_expected.equals(
        shape_actual
    ), f"Expected: {shape_expected}. Got: {shape_actual}"


def get_expected_shear_shape(length, width, shear_angle):
    dx = np.round(np.tan(np.deg2rad(shear_angle)) * width * 0.5, 3)
    x0 = 0
    x1 = length
    y0 = -width * 0.5
    y1 = width * 0.5
    return np.array(
        [
            [x0 - dx, y1],
            [x1 - dx, y1],
            [x1 + dx, y0],
            [x0 + dx, y0],
        ]
    )


@pytest.fixture
def shear_waveguide_symmetric():
    P = gf.path.straight(length=10)
    return gf.path.extrude(
        P, "strip", shear_angle_start=DEMO_PORT_ANGLE, shear_angle_end=DEMO_PORT_ANGLE
    )


@pytest.fixture
def shear_waveguide_start():
    P = gf.path.straight(length=10)
    return gf.path.extrude(
        P, "strip", shear_angle_start=DEMO_PORT_ANGLE, shear_angle_end=None
    )


@pytest.fixture
def shear_waveguide_end():
    P = gf.path.straight(length=10)
    return gf.path.extrude(
        P, "strip", shear_angle_start=None, shear_angle_end=DEMO_PORT_ANGLE
    )


@pytest.fixture
def regular_waveguide():
    P = gf.path.straight(length=10)
    return gf.path.extrude(P, "strip")


@pytest.fixture
def more_slanted_than_wide():
    P = gf.path.straight(length=0.1)
    return gf.path.extrude(P, "strip", shear_angle_start=60, shear_angle_end=60)


@pytest.fixture
def skinny():
    P = gf.path.straight(length=0.1)
    return gf.path.extrude(P, "strip")


@pytest.fixture
def test_mate_on_shear_xor_empty(
    regular_waveguide, shear_waveguide_start, shear_waveguide_end
) -> None:
    # two sheared components joined at the sheared port should appear the same as two straight component joined
    two_straights = gf.Component()
    c1 = two_straights << regular_waveguide
    c2 = two_straights << regular_waveguide
    c2.connect("o1", c1.ports["o2"])

    two_shears = gf.Component()
    c1 = two_shears << shear_waveguide_end
    c2 = two_shears << shear_waveguide_start
    c2.connect("o1", c1.ports["o2"])

    xor = gf.geometry.xor_diff(two_straights, two_shears)
    assert not xor.layers


def test_rotations_are_normal(
    regular_waveguide, shear_waveguide_start, shear_waveguide_end
) -> None:
    two_shears = gf.Component()
    c1 = two_shears << shear_waveguide_end
    c2 = two_shears << shear_waveguide_start
    c2.connect("o1", c1.ports["o2"])

    assert c2.rotation % 90 == 0


def test_area_stays_same(
    regular_waveguide,
    shear_waveguide_start,
    shear_waveguide_end,
    shear_waveguide_symmetric,
) -> None:
    components = [
        regular_waveguide,
        shear_waveguide_start,
        shear_waveguide_end,
        shear_waveguide_symmetric,
    ]
    areas = [c.area() for c in components]
    np.testing.assert_allclose(areas, desired=areas[0])


def test_area_stays_same_skinny(
    skinny,
    more_slanted_than_wide,
) -> None:
    components = [
        skinny,
        more_slanted_than_wide,
    ]
    areas = [c.area() for c in components]
    np.testing.assert_allclose(areas, desired=areas[0])


def test_mate_on_shear_xor_empty_transition() -> None:
    """two sheared components joined at the sheared port should appear the same
    as two straight component joined."""
    P = gf.path.straight(length=10)

    s = gf.Section(width=3, offset=0, layer=gf.LAYER.SLAB90, name="slab")
    X1 = gf.CrossSection(
        width=1,
        offset=0,
        layer=gf.LAYER.WG,
        name="core",
        port_names=("o1", "o2"),
        sections=[s],
    )
    s2 = gf.Section(width=2, offset=0, layer=gf.LAYER.SLAB90, name="slab")
    X2 = gf.CrossSection(
        width=0.5,
        offset=0,
        layer=gf.LAYER.WG,
        name="core",
        port_names=("o1", "o2"),
        sections=[s2],
    )
    t = gf.path.transition(X1, X2, width_type="linear")
    linear_taper = gf.path.extrude(P, t)

    P = gf.path.straight(length=10)

    s = gf.Section(width=3, offset=0, layer=gf.LAYER.SLAB90, name="slab")
    x1 = gf.CrossSection(
        width=1,
        offset=0,
        layer=gf.LAYER.WG,
        name="core",
        port_names=("o1", "o2"),
        sections=[s],
    )
    s2 = gf.Section(width=2, offset=0, layer=gf.LAYER.SLAB90, name="slab")
    x2 = gf.CrossSection(
        width=0.5,
        offset=0,
        layer=gf.LAYER.WG,
        name="core",
        port_names=("o1", "o2"),
        sections=[s2],
    )
    t = gf.path.transition(x1, x2, width_type="linear")
    linear_taper_sheared = gf.path.extrude(
        P, t, shear_angle_start=10, shear_angle_end=None
    )

    two_straights = gf.Component()
    c1 = two_straights << linear_taper
    c2 = two_straights << linear_taper
    c2.connect("o1", c1.ports["o1"])

    two_shears = gf.Component()
    c1 = two_shears << linear_taper_sheared
    c2 = two_shears << linear_taper_sheared
    c2.connect("o1", c1.ports["o1"])

    xor = gf.geometry.xor_diff(two_straights, two_shears)
    # xor.show()
    # two_straights.show()
    # two_shears.show()

    area = xor.area()
    assert area < 0.1, area


def test_mate_on_shear_xor_empty_curve() -> None:
    """two sheared components joined at the sheared port should appear the same
    as two straight component joined."""
    P = gf.path.euler()
    curve = gf.path.extrude(P, "strip")

    angle = 15
    P = gf.path.euler()
    curve_sheared1 = gf.path.extrude(P, "strip", shear_angle_end=angle)
    curve_sheared2 = gf.path.extrude(P, "strip", shear_angle_start=angle)

    two_straights = gf.Component()
    c1 = two_straights << curve
    c2 = two_straights << curve
    c2.connect("o1", c1.ports["o2"])

    two_shears = gf.Component()
    c1 = two_shears << curve_sheared1
    c2 = two_shears << curve_sheared2
    c2.connect("o1", c1.ports["o2"])

    xor = gf.geometry.xor_diff(two_straights, two_shears, precision=1e-2)
    assert not xor.layers, f"{xor.layers}"


def test_shear_angle_annotated_on_ports(
    shear_waveguide_start, shear_waveguide_end
) -> None:
    assert shear_waveguide_start.ports["o1"].shear_angle == DEMO_PORT_ANGLE
    assert shear_waveguide_start.ports["o2"].shear_angle is None

    assert shear_waveguide_end.ports["o2"].shear_angle == DEMO_PORT_ANGLE
    assert shear_waveguide_end.ports["o1"].shear_angle is None


def test_port_attributes(regular_waveguide, shear_waveguide_symmetric) -> None:
    regular_ports = [p.to_dict() for p in regular_waveguide.ports.values()]
    shear_ports = [p.to_dict() for p in shear_waveguide_symmetric.ports.values()]

    for p in shear_ports:
        shear_angle = p.pop("shear_angle")
        assert shear_angle == DEMO_PORT_ANGLE

    for p1, p2 in zip(regular_ports, shear_ports):
        for k in p.keys():
            assert p1[k] == p2[k], f"{k} differs! {p1[k]} != {p2[k]}"


def test_points_are_correct(shear_waveguide_symmetric):
    shear_angle = DEMO_PORT_ANGLE
    cs = gf.get_cross_section("strip")
    wg_width = cs.width
    length = 10
    points_expected = get_expected_shear_shape(
        length=length, width=wg_width, shear_angle=shear_angle
    )

    poly_actual = shear_waveguide_symmetric.get_polygons(by_spec=(1, 0))[0]
    assert_polygon_equals(points_expected, poly_actual)


def test_points_are_correct_wide():
    wg_width = 40
    length = 10
    P = gf.path.straight(length=length)
    shear_waveguide_symmetric = gf.path.extrude(
        P,
        {"cross_section": "strip", "settings": {"width": wg_width}},
        shear_angle_start=DEMO_PORT_ANGLE,
        shear_angle_end=DEMO_PORT_ANGLE,
    )

    shear_angle = DEMO_PORT_ANGLE

    points_expected = get_expected_shear_shape(
        length=length, width=wg_width, shear_angle=shear_angle
    )
    poly_actual = shear_waveguide_symmetric.get_polygons(by_spec=(1, 0))[0]
    assert_polygon_equals(points_expected, poly_actual)


def test_points_are_correct_short():
    wg_width = 40
    length = 0.5
    P = gf.path.straight(length=length)
    shear_waveguide_symmetric = gf.path.extrude(
        P,
        {"cross_section": "strip", "settings": {"width": wg_width}},
        shear_angle_start=DEMO_PORT_ANGLE,
        shear_angle_end=DEMO_PORT_ANGLE,
    )

    shear_angle = DEMO_PORT_ANGLE

    points_expected = get_expected_shear_shape(
        length=length, width=wg_width, shear_angle=shear_angle
    )
    poly_actual = shear_waveguide_symmetric.get_polygons(by_spec=(1, 0))[0]
    assert_polygon_equals(points_expected, poly_actual)


def test_points_are_correct_long():
    wg_width = 28
    length = 100
    P = gf.path.straight(length=length)
    shear_waveguide_symmetric = gf.path.extrude(
        P,
        {"cross_section": "strip", "settings": {"width": wg_width}},
        shear_angle_start=DEMO_PORT_ANGLE,
        shear_angle_end=DEMO_PORT_ANGLE,
    )

    shear_angle = DEMO_PORT_ANGLE

    points_expected = get_expected_shear_shape(
        length=length, width=wg_width, shear_angle=shear_angle
    )
    poly_actual = shear_waveguide_symmetric.get_polygons(by_spec=(1, 0))[0]
    assert_polygon_equals(points_expected, poly_actual)


def test_points_are_correct_multi_layer():
    length = 1000
    P = gf.path.straight(length=length)

    s = gf.Section(width=30, offset=0, layer=gf.LAYER.SLAB90, name="slab")
    X1 = gf.CrossSection(
        width=1,
        offset=0,
        layer=gf.LAYER.WG,
        name="core",
        port_names=("o1", "o2"),
        sections=[s],
    )
    shear_waveguide_symmetric = gf.path.extrude(
        P, X1, shear_angle_start=DEMO_PORT_ANGLE, shear_angle_end=DEMO_PORT_ANGLE
    )
    shear_angle = DEMO_PORT_ANGLE

    for layer, wg_width in [(gf.LAYER.WG, 1), (gf.LAYER.SLAB90, 30)]:
        points_expected = get_expected_shear_shape(
            length=length, width=wg_width, shear_angle=shear_angle
        )
        poly_actual = shear_waveguide_symmetric.get_polygons(by_spec=layer)[0]
        assert_polygon_equals(points_expected, poly_actual)


if __name__ == "__main__":
    test_mate_on_shear_xor_empty_curve()
    # P = gf.path.euler()
    # curve = gf.path.extrude(P, "strip")

    # angle = 15
    # P = gf.path.euler()
    # curve_sheared1 = gf.path.extrude(P, "strip", shear_angle_end=angle)
    # curve_sheared2 = gf.path.extrude(P, "strip", shear_angle_start=angle)

    # two_straights = gf.Component()
    # c1 = two_straights << curve
    # c2 = two_straights << curve
    # c2.connect("o1", c1.ports["o2"])

    # two_shears = gf.Component()
    # c1 = two_shears << curve_sheared1
    # c2 = two_shears << curve_sheared2
    # c2.connect("o1", c1.ports["o2"])

    # xor = gf.geometry.xor_diff(two_straights, two_shears, precision=1e-2)
    # assert not xor.layers, f"{xor.layers}"

    # two_straights.show()
    # two_shears.show()

    P = gf.path.euler()
    curve = gf.path.extrude(P, "strip")

    angle = 15
    P = gf.path.euler()
    curve_sheared1 = gf.path.extrude(P, "strip", shear_angle_end=angle)
    curve_sheared2 = gf.path.extrude(P, "strip", shear_angle_start=angle)

    two_straights = gf.Component()
    c1 = two_straights << curve
    c2 = two_straights << curve
    c2.connect("o1", c1.ports["o2"])

    two_shears = gf.Component()
    c1 = two_shears << curve_sheared1
    c2 = two_shears << curve_sheared2
    c2.connect("o1", c1.ports["o2"])

    xor = gf.geometry.xor_diff(two_straights, two_shears, precision=1e-2)
    xor.show()
