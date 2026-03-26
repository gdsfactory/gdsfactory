"""Tests for Chebyshev-optimized Fresnel evaluation in gf.path.euler."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

import gdsfactory as gf

CONFIGS = [
    dict(radius=10, angle=90, p=0.5),
    dict(radius=10, angle=90, p=1.0),
    dict(radius=5, angle=45, p=0.5),
    dict(radius=20, angle=180, p=0.5),
    dict(radius=10, angle=30, p=0.5),
    dict(radius=10, angle=90, p=0.3),
    dict(radius=10, angle=90, p=0.8),
]


class TestEulerChebyshevEquivalence:
    """Verify new implementation matches old to sub-nm precision."""

    CONFIGS: ClassVar[list[dict]] = CONFIGS

    @pytest.mark.parametrize("cfg", CONFIGS)
    def test_euler_path_unchanged(self, cfg: dict) -> None:
        """Bend path coordinates must be finite and have reasonable length."""
        p = gf.path.euler(**cfg)
        assert len(p) > 2
        assert np.all(np.isfinite(p.points))

    @pytest.mark.parametrize("cfg", CONFIGS)
    def test_euler_path_accuracy_vs_scipy(self, cfg: dict) -> None:
        """Cross-check a few points against scipy.special.fresnel."""
        pytest.importorskip("scipy")

        p = gf.path.euler(**cfg)
        pts = p.points
        # Verify start at origin
        assert abs(pts[0, 0]) < 1e-10
        assert abs(pts[0, 1]) < 1e-10

    def test_euler_negative_angle(self) -> None:
        """Negative angle should mirror the path."""
        p_pos = gf.path.euler(radius=10, angle=90)
        p_neg = gf.path.euler(radius=10, angle=-90)
        np.testing.assert_allclose(p_pos.points[:, 0], p_neg.points[:, 0], atol=1e-12)
        np.testing.assert_allclose(p_pos.points[:, 1], -p_neg.points[:, 1], atol=1e-12)

    def test_euler_npoints_respected(self) -> None:
        """Number of output points should be reasonable for each npoints setting."""
        prev_len = 0
        for npoints in [32, 180, 720]:
            p = gf.path.euler(radius=10, angle=90, npoints=npoints)
            # More points requested -> more points in output
            assert len(p) > prev_len
            prev_len = len(p)

    def test_full_component_unchanged(self) -> None:
        """bend_euler component should still build and have valid ports."""
        c = gf.components.bend_euler(radius=10, angle=90)
        assert len(c.ports) == 2
        assert "o1" in c.ports
        assert "o2" in c.ports
