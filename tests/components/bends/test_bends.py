import pytest

import gdsfactory as gf


def test_bend_circular_heater_min_radius() -> None:
    with pytest.raises(ValueError, match="min_bend_radius 1"):
        gf.components.bend_circular_heater(radius=1, allow_min_radius_violation=False)


def test_bend_circular() -> None:
    c = gf.components.bend_circular(radius=10, angle=90)
    assert c.info["length"] > 0
    assert c.info["radius"] == 10
    assert c.info["width"] > 0
    assert c.info["dy"] > 0
    assert len(c.ports) == 2

    with pytest.warns(UserWarning):
        c = gf.components.bend_circular(radius=10, angle=45)

    with pytest.raises(ValueError):
        gf.components.bend_circular(radius=1, allow_min_radius_violation=False)

    c = gf.components.bend_circular_all_angle(radius=10, angle=45)
    assert isinstance(c, gf.ComponentAllAngle)


def test_bend_circular_layer_width() -> None:
    c1 = gf.components.bend_circular(radius=10, layer=(2, 0), width=0.6)
    assert c1.info["width"] == 0.6
    assert (2, 0) in c1.layers

    c2 = gf.components.bend_circular(radius=10, layer=(3, 0))
    assert (3, 0) in c2.layers

    c3 = gf.components.bend_circular(radius=10, width=0.8)
    assert c3.info["width"] == 0.8

    c4 = gf.components.bend_circular(radius=10)
    assert c4.info["width"] > 0
    assert len(c4.layers) > 0


def test_bend_circular_allow_min_radius_violation() -> None:
    with pytest.raises(ValueError):
        gf.components.bend_circular(radius=1, allow_min_radius_violation=False)

    c = gf.components.bend_circular(radius=1, allow_min_radius_violation=True)
    assert c.info["radius"] == 1
    assert len(c.ports) == 2

    c = gf.components.bend_circular(radius=1, allow_min_radius_violation=True)
    assert c.info["radius"] == 1
    assert len(c.ports) == 2


def test_bend_euler() -> None:
    c1 = gf.components.bend_euler(radius=10, angle=90)
    assert isinstance(c1, gf.Component)
    assert not isinstance(c1, gf.ComponentAllAngle)

    c2 = gf.components.bend_euler_all_angle(radius=10, angle=90)
    assert isinstance(c2, gf.ComponentAllAngle)

    c3 = gf.components.bend_euler(radius=10, layer=(2, 0), width=0.5)
    assert isinstance(c3, gf.Component)
    assert c3.info["width"] == 0.5
    assert (2, 0) in c3.layers

    c4 = gf.components.bend_euler(radius=10, layer=(3, 0))
    assert isinstance(c4, gf.Component)
    assert (3, 0) in c4.layers

    c5 = gf.components.bend_euler(radius=10, width=0.8)
    assert isinstance(c5, gf.Component)
    assert c5.info["width"] == 0.8

    c6 = gf.components.bend_euler(radius=10)
    assert isinstance(c6, gf.Component)
    assert c6.info["width"] > 0
    assert len(c6.layers) > 0


def test_bend_euler_allow_min_radius_violation() -> None:
    with pytest.raises(ValueError):
        gf.components.bend_euler(radius=1, allow_min_radius_violation=False)

    c = gf.components.bend_euler(radius=1, allow_min_radius_violation=True)
    assert c.info["radius"] == 1
    assert len(c.ports) == 2


def test_bezier_curve() -> None:
    import numpy as np

    from gdsfactory.components.bends.bend_s import bezier_curve

    t = np.linspace(0, 1, 10)
    control_points = ((0, 0), (5, 0), (5, 2), (10, 2))
    points = bezier_curve(t, control_points)
    assert isinstance(points, np.ndarray)
    assert points.shape == (10, 2)
    assert np.allclose(points[0], (0, 0))
    assert np.allclose(points[-1], (10, 2))


def test_bezier() -> None:
    c = gf.components.bezier()
    assert isinstance(c, gf.Component)
    assert len(c.ports) == 2
    assert "length" in c.info
    assert "min_bend_radius" in c.info
    assert "start_angle" in c.info
    assert "end_angle" in c.info

    c2 = gf.components.bezier(
        control_points=((0, 0), (5, 0), (5, 2), (10, 2)),
        npoints=101,
        with_manhattan_facing_angles=False,
        start_angle=0,
        end_angle=0,
        cross_section="strip",
    )
    assert isinstance(c2, gf.Component)
    assert len(c2.ports) == 2


def test_find_min_curv_bezier_control_points() -> None:
    from gdsfactory.components.bends.bend_s import find_min_curv_bezier_control_points

    points = find_min_curv_bezier_control_points(
        start_point=(0, 0),
        end_point=(10, 5),
        start_angle=0,
        end_angle=45,
        npoints=101,
        alpha=0.05,
        nb_pts=2,
    )
    assert isinstance(points, tuple)
    assert len(points) == 4  # start + nb_pts + end
    assert all(isinstance(p, tuple) and len(p) == 2 for p in points)


def test_bend_s() -> None:
    c = gf.components.bend_s()
    assert isinstance(c, gf.Component)
    assert len(c.ports) == 2
    assert "length" in c.info
    assert "min_bend_radius" in c.info

    c2 = gf.components.bend_s(size=(10, 0))
    assert isinstance(c2, gf.Component)
    assert len(c2.ports) == 2
    assert c2.info["length"] == 10

    # Test with custom parameters
    c3 = gf.components.bend_s(
        size=(15, 5),
        npoints=150,
        cross_section="strip",
        allow_min_radius_violation=True,
    )
    assert isinstance(c3, gf.Component)
    assert len(c3.ports) == 2


def test_get_min_sbend_size() -> None:
    from gdsfactory.components.bends.bend_s import get_min_sbend_size

    size_x = get_min_sbend_size(size=(None, 10.0))
    assert isinstance(size_x, float)
    assert size_x > 0

    size_y = get_min_sbend_size(size=(10.0, None))
    assert isinstance(size_y, float)
    assert size_y > 0

    size = get_min_sbend_size(size=(None, 10.0), cross_section="strip", num_points=50)
    assert isinstance(size, float)
    assert size > 0

    from gdsfactory.cross_section import CrossSection

    custom_cross_section = CrossSection(radius=None)

    with pytest.raises(ValueError):
        get_min_sbend_size(size=(10.0, None), cross_section=custom_cross_section)

    with pytest.raises(ValueError):
        get_min_sbend_size(size=(10.0, 10.0))


def test_bezier_bend() -> None:
    from gdsfactory.components.bends.bend_s import bezier

    c = bezier()
    assert isinstance(c, gf.Component)
    assert len(c.ports) == 2

    c2 = bezier(control_points=((0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (6.0, 1.0)))
    assert isinstance(c2, gf.Component)
    assert len(c2.ports) == 2

    c3 = bezier(
        control_points=((0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (8.0, 2.0)),
        npoints=250,
        cross_section="strip",
        allow_min_radius_violation=True,
    )
    assert isinstance(c3, gf.Component)
    assert len(c3.ports) == 2

    c4 = bezier(
        control_points=(
            (0.0, 0.0),
            (5.0, 0.0),
            (5.0, 0.0),
            (10.0, 0.0),
        ),
        npoints=100,
    )
    assert isinstance(c4, gf.Component)
    assert len(c4.ports) == 2


if __name__ == "__main__":
    test_get_min_sbend_size()
