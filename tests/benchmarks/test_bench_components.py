"""Benchmarks for component creation and caching."""

from __future__ import annotations

import pytest

pytest.importorskip("pytest_benchmark")


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear the cell cache before each benchmark."""
    import kfactory as kf

    kf.kcl.clear_kcells()


def test_bench_straight_creation(benchmark: "pytest.BenchmarkFixture") -> None:
    """Benchmark creating a straight waveguide component."""
    import gdsfactory as gf

    def create_straight() -> gf.Component:
        gf.kcl.clear_kcells()
        return gf.components.straight(length=10)

    benchmark(create_straight)


def test_bench_mzi_creation(benchmark: "pytest.BenchmarkFixture") -> None:
    """Benchmark creating an MZI component (hierarchical)."""
    import gdsfactory as gf

    def create_mzi() -> gf.Component:
        gf.kcl.clear_kcells()
        return gf.components.mzis.mzi()

    benchmark(create_mzi)


def test_bench_bend_euler(benchmark: "pytest.BenchmarkFixture") -> None:
    """Benchmark creating an Euler bend."""
    import gdsfactory as gf

    def create_bend() -> gf.Component:
        gf.kcl.clear_kcells()
        return gf.components.bends.bend_euler(radius=10)

    benchmark(create_bend)
