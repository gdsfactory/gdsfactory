"""Benchmarks for path extrusion operations."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytest_benchmark")


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear the cell cache before each benchmark."""
    import kfactory as kf

    kf.kcl.clear_kcells()


def test_bench_path_extrude_simple(benchmark: "pytest.BenchmarkFixture") -> None:
    """Benchmark extruding a simple path."""
    import gdsfactory as gf

    def extrude_path() -> gf.Component:
        gf.kcl.clear_kcells()
        p = gf.path.straight(length=100)
        xs = gf.cross_section.cross_section(width=0.5)
        return p.extrude(xs)

    benchmark(extrude_path)


def test_bench_path_extrude_complex(benchmark: "pytest.BenchmarkFixture") -> None:
    """Benchmark extruding a complex multi-segment path."""
    import gdsfactory as gf

    def extrude_complex() -> gf.Component:
        gf.kcl.clear_kcells()
        points = np.array(
            [(0, 0), (100, 0), (100, 100), (200, 100), (200, 200)]
        )
        p = gf.Path(points)
        xs = gf.cross_section.cross_section(width=0.5)
        return p.extrude(xs)

    benchmark(extrude_complex)


def test_bench_path_euler(benchmark: "pytest.BenchmarkFixture") -> None:
    """Benchmark creating an euler path."""
    import gdsfactory as gf

    def euler_path() -> gf.Path:
        return gf.path.euler(radius=10, angle=90, p=0.5)

    benchmark(euler_path)
