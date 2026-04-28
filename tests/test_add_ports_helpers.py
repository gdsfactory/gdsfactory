"""Unit tests for extracted helper functions in add_ports.py."""

from __future__ import annotations

import pytest

from gdsfactory.add_ports import (
    _apply_inside_position,
    _infer_port_direction,
    _register_ports,
    _should_skip_marker,
    _snap_port_width,
)
from gdsfactory.port import Port

import gdsfactory as gf


# ---------------------------------------------------------------------------
# _should_skip_marker
# ---------------------------------------------------------------------------


class TestShouldSkipMarker:
    def test_skip_below_min_area(self) -> None:
        assert _should_skip_marker(1.0, 1.0, min_pin_area_um2=2.0, max_pin_area_um2=None, skip_square_ports=False)

    def test_skip_above_max_area(self) -> None:
        assert _should_skip_marker(10.0, 10.0, min_pin_area_um2=None, max_pin_area_um2=50.0, skip_square_ports=False)

    def test_skip_square_ports(self) -> None:
        assert _should_skip_marker(1.0, 1.0, min_pin_area_um2=None, max_pin_area_um2=None, skip_square_ports=True)

    def test_no_skip_rectangular(self) -> None:
        assert not _should_skip_marker(1.0, 2.0, min_pin_area_um2=None, max_pin_area_um2=None, skip_square_ports=True)

    def test_no_skip_within_bounds(self) -> None:
        assert not _should_skip_marker(2.0, 3.0, min_pin_area_um2=1.0, max_pin_area_um2=100.0, skip_square_ports=False)

    def test_no_skip_when_all_none(self) -> None:
        assert not _should_skip_marker(5.0, 5.0, min_pin_area_um2=None, max_pin_area_um2=None, skip_square_ports=False)

    def test_debug_prints_min_area(self, capsys: pytest.CaptureFixture[str]) -> None:
        _should_skip_marker(1.0, 1.0, min_pin_area_um2=2.0, max_pin_area_um2=None, skip_square_ports=False, debug=True)
        assert "skipping port" in capsys.readouterr().out

    def test_debug_prints_square(self, capsys: pytest.CaptureFixture[str]) -> None:
        _should_skip_marker(1.0, 1.0, min_pin_area_um2=None, max_pin_area_um2=None, skip_square_ports=True, debug=True)
        assert "skipping square port" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _infer_port_direction
# ---------------------------------------------------------------------------


class TestInferPortDirection:
    """Test port direction inference from marker geometry."""

    # Component bounds: 0..100 x 0..100, center at 50, 50
    BOUNDS = dict(xc=50.0, yc=50.0, dxmin=0.0, dymin=0.0, dxmax=100.0, dymax=100.0, tol=0.1)

    def test_east_rectangular(self) -> None:
        """Tall marker on the right side -> east (0 degrees)."""
        orient, width, x, y = _infer_port_direction(
            x=80.0, y=50.0, dx=2.0, dy=10.0,
            pxmin=79.0, pymin=45.0, pxmax=81.0, pymax=55.0,
            **self.BOUNDS,
        )
        assert orient == 0.0
        assert width == 10.0

    def test_west_rectangular(self) -> None:
        """Tall marker on the left side -> west (180 degrees)."""
        orient, width, x, y = _infer_port_direction(
            x=20.0, y=50.0, dx=2.0, dy=10.0,
            pxmin=19.0, pymin=45.0, pxmax=21.0, pymax=55.0,
            **self.BOUNDS,
        )
        assert orient == 180.0
        assert width == 10.0

    def test_north_rectangular(self) -> None:
        """Wide marker on the top side -> north (90 degrees)."""
        orient, width, x, y = _infer_port_direction(
            x=50.0, y=80.0, dx=10.0, dy=2.0,
            pxmin=45.0, pymin=79.0, pxmax=55.0, pymax=81.0,
            **self.BOUNDS,
        )
        assert orient == 90.0
        assert width == 10.0

    def test_south_rectangular(self) -> None:
        """Wide marker on the bottom side -> south (270 degrees)."""
        orient, width, x, y = _infer_port_direction(
            x=50.0, y=20.0, dx=10.0, dy=2.0,
            pxmin=45.0, pymin=19.0, pxmax=55.0, pymax=21.0,
            **self.BOUNDS,
        )
        assert orient == 270.0
        assert width == 10.0

    def test_square_east_boundary(self) -> None:
        """Square marker at east boundary -> east."""
        orient, width, _, _ = _infer_port_direction(
            x=99.95, y=50.0, dx=5.0, dy=5.0,
            pxmin=97.45, pymin=47.5, pxmax=100.0, pymax=52.5,
            **self.BOUNDS,
        )
        assert orient == 0.0

    def test_square_west_boundary(self) -> None:
        """Square marker at west boundary -> west."""
        orient, width, _, _ = _infer_port_direction(
            x=0.05, y=50.0, dx=5.0, dy=5.0,
            pxmin=0.0, pymin=47.5, pxmax=2.55, pymax=52.5,
            **self.BOUNDS,
        )
        assert orient == 180.0

    def test_square_north_boundary(self) -> None:
        """Square marker at north boundary -> north."""
        orient, width, _, _ = _infer_port_direction(
            x=50.0, y=99.95, dx=5.0, dy=5.0,
            pxmin=47.5, pymin=97.45, pxmax=52.5, pymax=100.0,
            **self.BOUNDS,
        )
        assert orient == 90.0

    def test_square_south_boundary(self) -> None:
        """Square marker at south boundary -> south."""
        orient, width, _, _ = _infer_port_direction(
            x=50.0, y=0.05, dx=5.0, dy=5.0,
            pxmin=47.5, pymin=0.0, pxmax=52.5, pymax=2.55,
            **self.BOUNDS,
        )
        assert orient == 270.0

    def test_square_fallback_east(self) -> None:
        """Square marker not at boundary but right of center -> east."""
        orient, width, _, _ = _infer_port_direction(
            x=60.0, y=50.0, dx=5.0, dy=5.0,
            pxmin=57.5, pymin=47.5, pxmax=62.5, pymax=52.5,
            **self.BOUNDS,
        )
        assert orient == 0.0

    def test_square_fallback_west(self) -> None:
        """Square marker not at boundary but left of center -> west."""
        orient, width, _, _ = _infer_port_direction(
            x=40.0, y=50.0, dx=5.0, dy=5.0,
            pxmin=37.5, pymin=47.5, pxmax=42.5, pymax=52.5,
            **self.BOUNDS,
        )
        assert orient == 180.0

    def test_ports_on_short_side(self) -> None:
        """With ports_on_short_side, a wide marker gives east/west."""
        orient, width, _, _ = _infer_port_direction(
            x=80.0, y=50.0, dx=10.0, dy=2.0,
            pxmin=75.0, pymin=49.0, pxmax=85.0, pymax=51.0,
            ports_on_short_side=True,
            **self.BOUNDS,
        )
        assert orient == 0.0
        assert width == 2.0


# ---------------------------------------------------------------------------
# _apply_inside_position
# ---------------------------------------------------------------------------


class TestApplyInsidePosition:
    BBOX = dict(pxmin=10.0, pymin=20.0, pxmax=30.0, pymax=40.0)

    def test_not_inside_returns_original(self) -> None:
        x, y = _apply_inside_position(0.0, 20.0, 30.0, **self.BBOX, inside=False)
        assert (x, y) == (20.0, 30.0)

    def test_east_inside(self) -> None:
        x, y = _apply_inside_position(0.0, 20.0, 30.0, **self.BBOX, inside=True)
        assert x == 30.0  # pxmax

    def test_west_inside(self) -> None:
        x, y = _apply_inside_position(180.0, 20.0, 30.0, **self.BBOX, inside=True)
        assert x == 10.0  # pxmin

    def test_north_inside(self) -> None:
        x, y = _apply_inside_position(90.0, 20.0, 30.0, **self.BBOX, inside=True)
        assert y == 40.0  # pymax

    def test_south_inside(self) -> None:
        x, y = _apply_inside_position(270.0, 20.0, 30.0, **self.BBOX, inside=True)
        assert y == 20.0  # pymin

    def test_east_inside_opposite(self) -> None:
        x, y = _apply_inside_position(0.0, 20.0, 30.0, **self.BBOX, inside=True, use_opposite_side=True)
        assert x == 10.0  # pxmin (opposite)

    def test_west_inside_opposite(self) -> None:
        x, y = _apply_inside_position(180.0, 20.0, 30.0, **self.BBOX, inside=True, use_opposite_side=True)
        assert x == 30.0  # pxmax (opposite)

    def test_north_inside_opposite(self) -> None:
        x, y = _apply_inside_position(90.0, 20.0, 30.0, **self.BBOX, inside=True, use_opposite_side=True)
        assert y == 20.0  # pymin (opposite)

    def test_south_inside_opposite(self) -> None:
        x, y = _apply_inside_position(270.0, 20.0, 30.0, **self.BBOX, inside=True, use_opposite_side=True)
        assert y == 40.0  # pymax (opposite)


# ---------------------------------------------------------------------------
# _snap_port_width
# ---------------------------------------------------------------------------


class TestSnapPortWidth:
    def test_zero_extra_width(self) -> None:
        result = _snap_port_width(0.5, 0.0)
        assert abs(result - 0.5) < 1e-9

    def test_with_extra_width(self) -> None:
        result = _snap_port_width(0.5, 0.1)
        assert abs(result - 0.4) < 1e-9

    def test_snaps_to_2nm(self) -> None:
        # 0.501 with no extra width -> round(0.501/0.002)*0.002 = round(250.5)*0.002 = 250*0.002 = 0.500
        result = _snap_port_width(0.501, 0.0)
        assert abs(result - 0.500) < 1e-9


# ---------------------------------------------------------------------------
# _register_ports
# ---------------------------------------------------------------------------


class TestRegisterPorts:
    def _make_port(self, name: str | None, x: float, orientation: float) -> Port:
        layer_idx = gf.get_layer((1, 0))
        return Port(name=name, center=(x, 0), width=0.5, orientation=orientation, layer=layer_idx)

    def test_adds_ports_to_component(self) -> None:
        c = gf.Component()
        ports = [self._make_port("o1", 0, 0), self._make_port("o2", 10, 180)]
        _register_ports(c, ports, auto_rename_ports=False)
        assert len(c.ports) == 2
        assert "o1" in [p.name for p in c.ports]
        assert "o2" in [p.name for p in c.ports]

    def test_raises_on_duplicate(self) -> None:
        c = gf.Component()
        layer_idx = gf.get_layer((1, 0))
        c.add_port(name="o1", center=(0, 0), width=0.5, orientation=0, layer=layer_idx)
        ports = [self._make_port("o1", 5, 0)]
        with pytest.raises(ValueError, match="already in"):
            _register_ports(c, ports, auto_rename_ports=False)

    def test_skips_none_names_when_allowed(self) -> None:
        c = gf.Component()
        ports = [self._make_port(None, 0, 0), self._make_port("o1", 10, 180)]
        _register_ports(c, ports, auto_rename_ports=False, allow_none_names=True)
        assert len(c.ports) == 1

    def test_auto_rename_ports(self) -> None:
        c = gf.Component()
        ports = [self._make_port("temp1", 0, 0), self._make_port("temp2", 10, 180)]
        _register_ports(c, ports, auto_rename_ports=True)
        assert len(c.ports) == 2
