from __future__ import annotations

import numpy as np
import pytest
import shapely.geometry as sg

import gdsfactory as gf

DEMO_PORT_ANGLE = 10


def assert_polygon_equals(coords_expected, coords_actual) -> None:
    coords_expected = gf.snap.snap_to_grid(coords_expected)
    coords_actual = gf.snap.snap_to_grid(coords_actual)
    shape_expected = sg.polygon.orient(sg.Polygon(coords_expected))
    shape_actual = sg.polygon.orient(sg.Polygon(coords_actual))

    assert shape_expected.equals(
        shape_actual
    ), f"Expected: {shape_expected}. Got: {shape_actual}"


def get_expected_shear_shape(length, width, shear_angle) -> np.ndarray:
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
def shear_waveguide_symmetric() -> gf.Component:
    P = gf.path.straight(length=10)
    return gf.path.extrude(
        P, "xs_sc", shear_angle_start=DEMO_PORT_ANGLE, shear_angle_end=DEMO_PORT_ANGLE
    )


@pytest.fixture
def shear_waveguide_start() -> gf.Component:
    P = gf.path.straight(length=10)
    return gf.path.extrude(
        P, "xs_sc", shear_angle_start=DEMO_PORT_ANGLE, shear_angle_end=None
    )


@pytest.fixture
def shear_waveguide_end() -> gf.Component:
    P = gf.path.straight(length=10)
    return gf.path.extrude(
        P, "xs_sc", shear_angle_start=None, shear_angle_end=DEMO_PORT_ANGLE
    )


@pytest.fixture
def regular_waveguide() -> gf.Component:
    P = gf.path.straight(length=10)
    return gf.path.extrude(P, "xs_sc")


@pytest.fixture
def more_slanted_than_wide() -> gf.Component:
    P = gf.path.straight(length=0.1)
    return gf.path.extrude(P, "xs_sc", shear_angle_start=60, shear_angle_end=60)


@pytest.fixture
def skinny() -> gf.Component:
    P = gf.path.straight(length=0.1)
    return gf.path.extrude(P, "xs_sc")


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

    s0 = gf.Section(width=1, offset=0, layer=(1, 0), port_names=("o1", "o2"))
    s1 = gf.Section(width=3, offset=0, layer=(3, 0), name="slab")
    X1 = gf.CrossSection(sections=(s0, s1))

    s0 = gf.Section(width=0.5, offset=0, layer=(1, 0), port_names=("o1", "o2"))
    s1 = gf.Section(width=2, offset=0, layer=(3, 0), name="slab")
    X2 = gf.CrossSection(sections=(s0, s1))
    t = gf.path.transition(X1, X2, width_type="linear")
    linear_taper = gf.path.extrude_transition(P, t)

    linear_taper_sheared = gf.path.extrude_transition(
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
    area = xor.area()
    assert area < 0.1, area


def test_mate_on_shear_xor_empty_curve() -> None:
    """two sheared components joined at the sheared port should appear the same
    as two straight component joined."""
    P = gf.path.euler()
    curve = gf.path.extrude(P, "xs_sc")

    angle = 15
    P = gf.path.euler()
    curve_sheared1 = gf.path.extrude(P, "xs_sc", shear_angle_end=angle)
    curve_sheared2 = gf.path.extrude(P, "xs_sc", shear_angle_start=angle)

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


def test_points_are_correct(shear_waveguide_symmetric) -> None:
    shear_angle = DEMO_PORT_ANGLE
    cs = gf.get_cross_section("xs_sc")
    wg_width = cs.width
    length = 10
    points_expected = get_expected_shear_shape(
        length=length, width=wg_width, shear_angle=shear_angle
    )

    layer = (1, 0)
    poly_actual = shear_waveguide_symmetric.get_polygons(by_spec=layer)[0]
    assert_polygon_equals(points_expected, poly_actual)


def test_points_are_correct_wide() -> None:
    wg_width = 40
    length = 10
    P = gf.path.straight(length=length)
    shear_waveguide_symmetric = gf.path.extrude(
        p=P,
        cross_section={"cross_section": "xs_sc", "settings": {"width": wg_width}},
        shear_angle_start=DEMO_PORT_ANGLE,
        shear_angle_end=DEMO_PORT_ANGLE,
    )

    shear_angle = DEMO_PORT_ANGLE

    points_expected = get_expected_shear_shape(
        length=length, width=wg_width, shear_angle=shear_angle
    )
    layer = (1, 0)
    poly_actual = shear_waveguide_symmetric.get_polygons(by_spec=layer)[0]
    assert_polygon_equals(points_expected, poly_actual)


def test_points_are_correct_short() -> None:
    wg_width = 40
    length = 0.5
    P = gf.path.straight(length=length)
    shear_waveguide_symmetric = gf.path.extrude(
        P,
        {"cross_section": "xs_sc", "settings": {"width": wg_width}},
        shear_angle_start=DEMO_PORT_ANGLE,
        shear_angle_end=DEMO_PORT_ANGLE,
    )

    shear_angle = DEMO_PORT_ANGLE

    points_expected = get_expected_shear_shape(
        length=length, width=wg_width, shear_angle=shear_angle
    )
    layer = (1, 0)
    poly_actual = shear_waveguide_symmetric.get_polygons(by_spec=layer)[0]
    assert_polygon_equals(points_expected, poly_actual)


def test_points_are_correct_long() -> None:
    wg_width = 28
    length = 100
    P = gf.path.straight(length=length)
    shear_waveguide_symmetric = gf.path.extrude(
        P,
        {"cross_section": "xs_sc", "settings": {"width": wg_width}},
        shear_angle_start=DEMO_PORT_ANGLE,
        shear_angle_end=DEMO_PORT_ANGLE,
    )

    shear_angle = DEMO_PORT_ANGLE

    points_expected = get_expected_shear_shape(
        length=length, width=wg_width, shear_angle=shear_angle
    )
    layer = (1, 0)
    poly_actual = shear_waveguide_symmetric.get_polygons(by_spec=layer)[0]
    assert_polygon_equals(points_expected, poly_actual)


def test_points_are_correct_multi_layer() -> None:
    length = 1000
    P = gf.path.straight(length=length)

    s0 = gf.Section(width=1, offset=0, layer=(1, 0), port_names=("o1", "o2"))
    s1 = gf.Section(width=30, offset=0, layer=(3, 0), name="slab")

    X1 = gf.CrossSection(sections=(s0, s1))

    shear_waveguide_symmetric = gf.path.extrude(
        P, X1, shear_angle_start=DEMO_PORT_ANGLE, shear_angle_end=DEMO_PORT_ANGLE
    )
    shear_angle = DEMO_PORT_ANGLE
    shear_waveguide_symmetric.show()

    for layer, wg_width in [((1, 0), 1), ((3, 0), 30)]:
        points_expected = get_expected_shear_shape(
            length=length, width=wg_width, shear_angle=shear_angle
        )
        poly_actual = shear_waveguide_symmetric.get_polygons(by_spec=layer)[0]
        assert_polygon_equals(points_expected, poly_actual)


if __name__ == "__main__":
    # test_mate_on_shear_xor_empty_transition()
    test_points_are_correct_multi_layer()
