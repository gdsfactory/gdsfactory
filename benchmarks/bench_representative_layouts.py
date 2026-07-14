from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import numpy as np

import gdsfactory as gf

ComponentFactory = Callable[[], gf.Component]

gf.gpdk.PDK.activate()


def build_many_instances_component(count: int = 5000) -> gf.Component:
    """One cell containing many references to the same straight."""
    gf.clear_cache()
    component = gf.Component()
    straight = gf.components.straight(length=100, width=0.5)
    columns = math.ceil(math.sqrt(count))
    pitch_x = 120.0
    pitch_y = 2.0

    for index in range(count):
        ref = component.add_ref(straight)
        ref.dmove(((index % columns) * pitch_x, (index // columns) * pitch_y))

    return component


def build_many_unique_cells_component(count: int = 2000) -> gf.Component:
    """One top cell referencing many unique straight cells."""
    gf.clear_cache()
    component = gf.Component()
    columns = math.ceil(math.sqrt(count))
    pitch_x = 40.0
    pitch_y = 2.0

    for index in range(count):
        straight = gf.components.straight(length=10.0 + index * 0.01, width=0.5)
        ref = component.add_ref(straight)
        ref.dmove(((index % columns) * pitch_x, (index // columns) * pitch_y))

    return component


def build_many_trivial_polygons_component(count: int = 2000) -> gf.Component:
    """One cell containing many simple box polygons."""
    gf.clear_cache()
    component = gf.Component()
    columns = math.ceil(math.sqrt(count))
    size = 5.0
    pitch = 8.0

    for index in range(count):
        x = (index % columns) * pitch
        y = (index // columns) * pitch
        component.add_polygon(
            [(x, y), (x + size, y), (x + size, y + size), (x, y + size)],
            layer=(1, 0),
        )

    return component


def build_complex_polygons_component(
    count: int = 50, points_per_polygon: int = 10_000
) -> gf.Component:
    """One cell containing a few polygons with many points."""
    gf.clear_cache()
    component = gf.Component()
    columns = math.ceil(math.sqrt(count))
    theta = np.linspace(0, 2 * np.pi, points_per_polygon, endpoint=False)
    radius = 1 + 0.05 * np.sin(17 * theta) + 0.02 * np.sin(41 * theta)
    unit_polygon = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))

    for index in range(count):
        scale = 10.0 + (index % 5)
        polygon = unit_polygon * scale
        polygon[:, 0] += (index % columns) * 40.0
        polygon[:, 1] += (index // columns) * 40.0
        component.add_polygon(polygon, layer=(1, 0))

    return component


REPRESENTATIVE_LAYOUT_FACTORIES: dict[str, ComponentFactory] = {
    "instances_5000": build_many_instances_component,
    "unique_cells_2000": build_many_unique_cells_component,
    "trivial_polygons_2000": build_many_trivial_polygons_component,
    "complex_polygons_50x10000": build_complex_polygons_component,
}

WORKLOADS = tuple(REPRESENTATIVE_LAYOUT_FACTORIES)


class TemporaryGdsPath:
    def __init__(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.path = Path(self._tmpdir.name) / "component.gds"

    def cleanup(self) -> None:
        self._tmpdir.cleanup()


class TimeRepresentativeLayoutBuild:
    param_names: ClassVar[tuple[str, ...]] = ("workload",)
    params: ClassVar[tuple[tuple[str, ...], ...]] = (WORKLOADS,)

    def setup(self, workload: str) -> None:
        self.factory = REPRESENTATIVE_LAYOUT_FACTORIES[workload]

    def time_build_layout(self, workload: str) -> None:
        self.factory()


class TimeRepresentativeLayoutWriteGds:
    param_names: ClassVar[tuple[str, ...]] = ("workload",)
    params: ClassVar[tuple[tuple[str, ...], ...]] = (WORKLOADS,)

    def setup(self, workload: str) -> None:
        self.component = REPRESENTATIVE_LAYOUT_FACTORIES[workload]()
        self.gds_path = TemporaryGdsPath()

    def teardown(self, workload: str) -> None:
        self.gds_path.cleanup()

    def time_write_gds(self, workload: str) -> None:
        self.component.write_gds(gdspath=self.gds_path.path, with_metadata=False)
