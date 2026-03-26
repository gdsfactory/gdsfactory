#!/usr/bin/env python3
"""Benchmark: old (Taylor) vs new (Chebyshev) Fresnel evaluation for euler bends.

Self-contained — embeds both implementations inline so reviewers can run it
on the unmodified repo.  Dependencies: numpy (required), mpmath (optional,
for accuracy section only).

Usage:
    python benchmarks/bench_euler_fresnel.py
"""

from __future__ import annotations

import math
import time

import numpy as np

# ---------------------------------------------------------------------------
# Old implementation (Taylor with per-call factorial)
# ---------------------------------------------------------------------------


def _fresnel_old(R0: float, s: float, num_pts: int, n_iter: int = 8):
    """Original Taylor-series Fresnel with factorial recomputation."""
    t = np.linspace(0, s / float(np.sqrt(2) * R0), num_pts)
    n = np.arange(n_iter)
    exp = np.array([4 * n + 1, 4 * n + 3])
    den = np.empty(shape=(2, n_iter))
    den[0] = [math.factorial(2 * i) * (4 * i + 1) for i in n]
    den[1] = [math.factorial(2 * i + 1) * (4 * i + 3) for i in n]
    den *= (-1.0) ** n
    series = (t ** exp[..., None] / den[..., None]).sum(axis=1)
    return np.sqrt(2) * R0 * series


# Old implementation with precomputed coefficients (isolate precompute win)
_OLD_COEFFS_CACHE: dict[int, tuple] = {}


def _fresnel_old_precomputed(R0: float, s: float, num_pts: int, n_iter: int = 8):
    """Taylor-series Fresnel with cached coefficients (precomputed)."""
    t = np.linspace(0, s / float(np.sqrt(2) * R0), num_pts)
    if n_iter not in _OLD_COEFFS_CACHE:
        n = np.arange(n_iter)
        exp = np.array([4 * n + 1, 4 * n + 3])
        den = np.empty(shape=(2, n_iter))
        den[0] = [math.factorial(2 * i) * (4 * i + 1) for i in n]
        den[1] = [math.factorial(2 * i + 1) * (4 * i + 3) for i in n]
        den *= (-1.0) ** n
        _OLD_COEFFS_CACHE[n_iter] = (exp, den)
    exp, den = _OLD_COEFFS_CACHE[n_iter]
    series = (t ** exp[..., None] / den[..., None]).sum(axis=1)
    return np.sqrt(2) * R0 * series


# ---------------------------------------------------------------------------
# New implementation (Chebyshev)
# ---------------------------------------------------------------------------

_EULER_F: list[float] = [
    0.99999998960479,
    -0.09999993603363509,
    0.004629565266313682,
    -0.00010681326037385146,
    1.4545123517204771e-06,
    -1.2709937461472671e-08,
    6.392886959909195e-11,
]
_EULER_G: list[float] = [
    0.3333333326687886,
    -0.023809519722479022,
    0.0007575716481163624,
    -1.322596032033227e-05,
    1.4475800740514079e-07,
    -1.0630106497978158e-09,
    4.710134391674656e-12,
]


def _fresnel_new(R0: float, s: float, num_pts: int, n_iter: int = 8):
    """Chebyshev-optimised Fresnel with Horner evaluation."""
    t = np.linspace(0, s / float(np.sqrt(2) * R0), num_pts)
    t2 = t * t
    t4 = t2 * t2

    nf = len(_EULER_F)
    fv = np.full_like(t, _EULER_F[nf - 1])
    gv = np.full_like(t, _EULER_G[nf - 1])
    for i in range(nf - 2, -1, -1):
        fv = fv * t4 + _EULER_F[i]
        gv = gv * t4 + _EULER_G[i]

    x = t * fv
    y = t * t2 * gv
    return np.sqrt(2) * R0 * np.array([x, y])


# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        "label": "720 pts (default)",
        "R0": 1.0,
        "s_factor": 0.5,
        "angle": 90,
        "npoints": 720,
    },
    {"label": "180 pts (GDS)", "R0": 1.0, "s_factor": 1.0, "angle": 90, "npoints": 180},
    {"label": "32 pts (WebGL)", "R0": 1.0, "s_factor": 0.5, "angle": 90, "npoints": 32},
    {
        "label": "720 pts (U-bend)",
        "R0": 1.0,
        "s_factor": 0.5,
        "angle": 180,
        "npoints": 720,
    },
]

N_TIMING = 2000


def compute_s(R0, angle_deg, p):
    alpha = np.radians(angle_deg)
    return float(R0 * np.sqrt(p * alpha))


def bench():
    print("=" * 76)
    print("Fresnel evaluation benchmark: Taylor vs Chebyshev")
    print("=" * 76)
    print()

    header = f"{'Config':<22} {'Old (µs)':>10} {'Precomp (µs)':>13} {'New (µs)':>10} {'Speedup':>8} {'Max dev (nm)':>13}"
    print(header)
    print("-" * len(header))

    for cfg in CONFIGS:
        R0 = cfg["R0"]
        s = compute_s(R0, cfg["angle"], cfg["s_factor"])
        npts = cfg["npoints"]

        # Warm up
        _fresnel_old(R0, s, npts)
        _fresnel_old_precomputed(R0, s, npts)
        _fresnel_new(R0, s, npts)

        # Time old
        t0 = time.perf_counter()
        for _ in range(N_TIMING):
            _fresnel_old(R0, s, npts)
        t_old = (time.perf_counter() - t0) / N_TIMING * 1e6

        # Time precomputed Taylor
        t0 = time.perf_counter()
        for _ in range(N_TIMING):
            _fresnel_old_precomputed(R0, s, npts)
        t_pre = (time.perf_counter() - t0) / N_TIMING * 1e6

        # Time new
        t0 = time.perf_counter()
        for _ in range(N_TIMING):
            _fresnel_new(R0, s, npts)
        t_new = (time.perf_counter() - t0) / N_TIMING * 1e6

        # Accuracy: deviation between old and new (in nm, assuming R=10um)
        old = _fresnel_old(R0, s, npts)
        new = _fresnel_new(R0, s, npts)
        # Scale to 10 um radius
        scale_nm = 10_000  # 10 um = 10000 nm
        max_dev = np.max(np.abs(old - new)) * scale_nm

        speedup = t_old / t_new
        print(
            f"{cfg['label']:<22} {t_old:>10.1f} {t_pre:>13.1f} {t_new:>10.1f} {speedup:>7.1f}x {max_dev:>12.4f}"
        )

    print()

    # ---------------------------------------------------------------------------
    # Accuracy vs mpmath reference
    # ---------------------------------------------------------------------------
    print("Accuracy vs mpmath 50-digit reference (10 µm radius bend)")
    print("-" * 60)
    try:
        import mpmath

        mpmath.mp.dps = 50

        def fresnel_ref(t_val):
            t = mpmath.mpf(t_val)
            x = sum(
                mpmath.mpf(-1) ** n
                * t ** (4 * n + 1)
                / (mpmath.factorial(2 * n) * (4 * n + 1))
                for n in range(30)
            )
            y = sum(
                mpmath.mpf(-1) ** n
                * t ** (4 * n + 3)
                / (mpmath.factorial(2 * n + 1) * (4 * n + 3))
                for n in range(30)
            )
            return float(x), float(y)

        for cfg in CONFIGS:
            R0 = cfg["R0"]
            s = compute_s(R0, cfg["angle"], cfg["s_factor"])
            npts = min(cfg["npoints"], 500)

            t_arr = np.linspace(0, s / float(np.sqrt(2) * R0), npts)
            new = _fresnel_new(R0, s, npts)

            scale = np.sqrt(2) * R0
            max_err = 0.0
            for j, tv in enumerate(t_arr):
                xr, yr = fresnel_ref(tv)
                ex = abs(new[0, j] - scale * xr)
                ey = abs(new[1, j] - scale * yr)
                max_err = max(max_err, ex, ey)

            # max_err is in R0-normalised units; scale to 10 um bend in nm
            max_err_nm = max_err * 10_000
            print(f"  {cfg['label']:<22}  max error = {max_err_nm:.6f} nm")

    except ImportError:
        print("  (mpmath not installed — skipping accuracy section)")

    print()
    print("Done.")


if __name__ == "__main__":
    bench()
