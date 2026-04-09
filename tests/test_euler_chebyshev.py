"""Tests for scipy-based Fresnel evaluation in gf.path.euler."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
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


class TestEulerFresnelAccuracy:
    """Verify euler path implementation against scipy reference."""

    CONFIGS: ClassVar[list[dict]] = CONFIGS

    @pytest.mark.parametrize("cfg", CONFIGS)
    def test_euler_path_unchanged(self, cfg: dict) -> None:
        """Bend path coordinates must be finite and have reasonable length."""
        p = gf.path.euler(**cfg)
        assert len(p) > 2
        assert np.all(np.isfinite(p.points))

    def test_fresnel_vs_scipy_reference(self) -> None:
        """Cross-check _fresnel output against direct scipy.special.fresnel call."""
        R0, s, num_pts = 1.0, 0.886, 500  # R0=1, s for 90deg p=0.5
        result = _fresnel(R0, s, num_pts)
        x, y = result[0], result[1]

        # Recompute the same t array and evaluate scipy directly
        t = np.linspace(0, s / float(np.sqrt(2) * R0), num_pts)
        sqrt_half_pi = np.sqrt(np.pi / 2)
        sqrt_2_over_pi = np.sqrt(2 / np.pi)
        S, C = scipy_fresnel(t * sqrt_2_over_pi)
        ref_x = np.sqrt(2) * R0 * C * sqrt_half_pi
        ref_y = np.sqrt(2) * R0 * S * sqrt_half_pi

        np.testing.assert_allclose(x, ref_x, atol=1e-14)
        np.testing.assert_allclose(y, ref_y, atol=1e-14)

    def test_fresnel_angular_vs_scipy_reference(self) -> None:
        """Cross-check _fresnel_angular output against direct scipy call."""
        R0, s, num_pts = 1.0, 0.886, 500
        result = _fresnel_angular(R0, s, num_pts)
        x, y = result[0], result[1]

        t_max = s / float(np.sqrt(2) * R0)
        thetas = np.linspace(0, t_max**2 / 2, num_pts)
        t = np.sqrt(2 * thetas)
        sqrt_half_pi = np.sqrt(np.pi / 2)
        sqrt_2_over_pi = np.sqrt(2 / np.pi)
        S, C = scipy_fresnel(t * sqrt_2_over_pi)
        ref_x = np.sqrt(2) * R0 * C * sqrt_half_pi
        ref_y = np.sqrt(2) * R0 * S * sqrt_half_pi

        np.testing.assert_allclose(x, ref_x, atol=1e-14)
        np.testing.assert_allclose(y, ref_y, atol=1e-14)

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
