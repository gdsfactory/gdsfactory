"""Benchmarks for GDS I/O operations."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("pytest_benchmark")


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear the cell cache before each benchmark."""
    import kfactory as kf

    kf.kcl.clear_kcells()


@pytest.fixture
def sample_gds(tmp_path: Path) -> Path:
    """Create a sample GDS file for read benchmarks."""
    import gdsfactory as gf

    c = gf.components.mzis.mzi()
    filepath = tmp_path / "bench_sample.gds"
    c.write_gds(filepath)
    gf.kcl.clear_kcells()
    return filepath


def test_bench_write_gds(benchmark: "pytest.BenchmarkFixture") -> None:
    """Benchmark writing a component to GDS."""
    import gdsfactory as gf

    c = gf.components.mzis.mzi()

    def write_gds() -> None:
        with tempfile.NamedTemporaryFile(suffix=".gds") as f:
            c.write_gds(f.name)

    benchmark(write_gds)


def test_bench_read_gds(
    benchmark: "pytest.BenchmarkFixture", sample_gds: Path
) -> None:
    """Benchmark reading a GDS file."""
    import gdsfactory as gf

    def read_gds() -> gf.Component:
        gf.kcl.clear_kcells()
        return gf.import_gds(sample_gds)

    benchmark(read_gds)
