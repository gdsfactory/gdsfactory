"""Tests for Euler-path Chebyshev-related behavior, including Fresnel cross-checks."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import fresnel as scipy_fresnel

import gdsfactory as gf
from gdsfactory.path import _fresnel, _fresnel_angular

CONFIGS = [
    dict(radius=10, angle=90, p=0.5),
    dict(radius=10, angle=90, p=1.0),
    dict(radius=5, angle=45, p=0.5),
    dict(radius=20, angle=180, p=0.5),
    dict(radius=10, angle=30, p=0.5),
    dict(radius=10, angle=90, p=0.3),
    dict(radius=10, angle=90, p=0.8),
]


def _compute_t_max(s: float, R0: float) -> float:
    """Return shared upper bound for Fresnel parameterization."""
    return s / float(np.sqrt(2) * R0)


class TestEulerFresnelAccuracy:
    """Verify euler path implementation against scipy reference."""

    CONFIGS: ClassVar[list[dict]] = CONFIGS
    SQRT_HALF_PI: ClassVar[float] = np.sqrt(np.pi / 2)
    SQRT_2_OVER_PI: ClassVar[float] = np.sqrt(2 / np.pi)

    @given(
        radius=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
        angle=st.floats(min_value=1, max_value=180, allow_nan=False, allow_infinity=False),
        p=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_euler_path_finite_points(self, radius: float, angle: float, p: float) -> None:
        """Bend path coordinates must be finite and have reasonable length."""
        path = gf.path.euler(radius=radius, angle=angle, p=p)
        assert len(path) >= 2
        assert np.all(np.isfinite(path.points))

    def test_fresnel_vs_scipy_reference(self) -> None:
        """Cross-check _fresnel output against direct scipy.special.fresnel call."""
        R0, num_pts = 1.0, 500
        # For a 90-degree bend with p=0.5, the Euler parameter is:
        # s = R0 * sqrt(p * alpha), with alpha in radians.
        # alpha = pi/2 -> s ≈ 0.886226925 for R0=1.
        p = 0.5
        alpha = np.deg2rad(90.0)
        s = R0 * np.sqrt(p * alpha)
        result = _fresnel(R0, s, num_pts)
        x, y = result[0], result[1]

        # Recompute the same t array and evaluate scipy directly
        t_max = _compute_t_max(s, R0)
        t = np.linspace(0, t_max, num_pts)
        S, C = scipy_fresnel(t * self.SQRT_2_OVER_PI)
        ref_x = np.sqrt(2) * R0 * C * self.SQRT_HALF_PI
        ref_y = np.sqrt(2) * R0 * S * self.SQRT_HALF_PI

        np.testing.assert_allclose(x, ref_x, atol=1e-14)
        np.testing.assert_allclose(y, ref_y, atol=1e-14)

    def test_fresnel_angular_vs_scipy_reference(self) -> None:
        """Cross-check _fresnel_angular output against direct scipy call."""
        R0, num_pts = 1.0, 500
        # For a 90-degree bend with p=0.5, use s = R0 * sqrt(p * alpha)
        # with alpha in radians (alpha = pi/2), so s ≈ 0.886226925 at R0=1.
        p = 0.5
        alpha = np.deg2rad(90.0)
        s = R0 * np.sqrt(p * alpha)
        result = _fresnel_angular(R0, s, num_pts)
        x, y = result[0], result[1]

        t_max = _compute_t_max(s, R0)
        thetas = np.linspace(0, t_max**2 / 2, num_pts)
        t = np.sqrt(2 * thetas)
        S, C = scipy_fresnel(t * self.SQRT_2_OVER_PI)
        ref_x = np.sqrt(2) * R0 * C * self.SQRT_HALF_PI
        ref_y = np.sqrt(2) * R0 * S * self.SQRT_HALF_PI

        np.testing.assert_allclose(x, ref_x, atol=1e-14)
        np.testing.assert_allclose(y, ref_y, atol=1e-14)

    @given(
        radius=st.floats(min_value=1, max_value=50, allow_nan=False, allow_infinity=False),
        angle=st.floats(min_value=10, max_value=180, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_euler_negative_angle(self, radius: float, angle: float) -> None:
        """Negative angle should mirror the path."""
        p_pos = gf.path.euler(radius=radius, angle=angle)
        p_neg = gf.path.euler(radius=radius, angle=-angle)
        np.testing.assert_allclose(p_pos.points[:, 0], p_neg.points[:, 0], atol=1e-12)
        np.testing.assert_allclose(p_pos.points[:, 1], -p_neg.points[:, 1], atol=1e-12)

    @given(
        npoints1=st.integers(min_value=16, max_value=200),
        npoints2=st.integers(min_value=201, max_value=1000),
    )
    @settings(max_examples=50)
    def test_euler_npoints_respected(self, npoints1: int, npoints2: int) -> None:
        """More requested points should yield more output points."""
        p1 = gf.path.euler(radius=10, angle=90, npoints=npoints1)
        p2 = gf.path.euler(radius=10, angle=90, npoints=npoints2)
        assert len(p2) > len(p1)

    def test_full_component_unchanged(self) -> None:
        """bend_euler component should still build and have valid ports."""
        c = gf.components.bend_euler(radius=10, angle=90)
        assert len(c.ports) == 2
        assert "o1" in c.ports
        assert "o2" in c.ports
