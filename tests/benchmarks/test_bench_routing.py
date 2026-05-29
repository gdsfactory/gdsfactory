"""Benchmarks for routing operations."""

from __future__ import annotations

import pytest

pytest.importorskip("pytest_benchmark")


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear the cell cache before each benchmark."""
    import kfactory as kf

    kf.kcl.clear_kcells()


def test_bench_route_single(benchmark: "pytest.BenchmarkFixture") -> None:
    """Benchmark routing a single connection."""
    import gdsfactory as gf

    def route_single() -> gf.Component:
        gf.kcl.clear_kcells()
        c = gf.Component()
        s1 = c << gf.components.straight(length=10)
        s2 = c << gf.components.straight(length=10)
        s2.d.move((100, 50))
        gf.routing.route_bundle(
            c,
            ports1=[s1.ports["o2"]],
            ports2=[s2.ports["o1"]],
        )
        return c

    benchmark(route_single)


def test_bench_route_bundle_10(benchmark: "pytest.BenchmarkFixture") -> None:
    """Benchmark bundle routing with 10 ports."""
    import gdsfactory as gf

    def route_bundle() -> gf.Component:
        gf.kcl.clear_kcells()
        c = gf.Component()
        columns_left = 2
        columns_right = 2

        left = c << gf.components.pads.pad_array(
            columns=columns_left, rows=5, port_orientation=270
        )
        right = c << gf.components.pads.pad_array(
            columns=columns_right, rows=5, port_orientation=90
        )
        right.d.move((500, 0))

        ports_left = left.ports.filter(orientation=270)
        ports_right = right.ports.filter(orientation=90)

        gf.routing.route_bundle(
            c,
            ports1=ports_left,
            ports2=ports_right,
            cross_section="metal_routing",
        )
        return c

    benchmark(route_bundle)
