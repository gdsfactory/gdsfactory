from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import gdsfactory as gf
from gdsfactory.export.to_np import to_np
from gdsfactory.labels.write_labels import write_labels
from gdsfactory.read.import_gds import import_gds
from gdsfactory.samples.big_device import big_device as optical_big_device
from gdsfactory.samples.big_device_electrical import (
    big_device as electrical_big_device,
)
from gdsfactory.samples.coh_rx_single_pol import coh_rx_single_pol
from gdsfactory.samples.coh_tx_dual_pol import coh_tx_dual_pol
from gdsfactory.samples.sample_reticle import sample_reticle
from gdsfactory.samples.sample_reticle_with_labels import (
    sample_reticle_with_labels,
)

ComponentFactory = Callable[[], gf.Component]

gf.gpdk.PDK.activate()


class TemporaryGdsPath:
    def __init__(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.path = Path(self._tmpdir.name) / "component.gds"

    def cleanup(self) -> None:
        self._tmpdir.cleanup()


class TemporaryCsvPath:
    def __init__(self) -> None:
        self._tmpdir = TemporaryDirectory()
        self.path = Path(self._tmpdir.name) / "labels.csv"

    def cleanup(self) -> None:
        self._tmpdir.cleanup()


def build_mzi_fiber_array() -> gf.Component:
    """MZI with routed grating-coupler IO."""
    gf.clear_cache()
    component = gf.components.mzi(delta_length=100, length_x=200)
    return gf.routing.add_fiber_array(component)


def build_big_device_fiber_array() -> gf.Component:
    """High-port optical block routed to a fiber array."""
    gf.clear_cache()
    component = optical_big_device(nports=10)
    return gf.routing.add_fiber_array(
        component=component,
        radius=5.0,
        fanout_length=50.0,
        radius_loopback=10,
    )


def build_big_device_electrical_pads() -> gf.Component:
    """High-port electrical block routed to top pads."""
    gf.clear_cache()
    component = electrical_big_device(nports=32)
    top_port_names = tuple(port.name for port in component.ports.filter(orientation=90))
    return gf.routing.add_pads_top(
        component,
        port_names=top_port_names,
        fanout_length=None,
        pad_pitch=120,
        straight_separation=20,
    )


def build_lidar_with_pads(elements: int = 16) -> gf.Component:
    """Phased-array style splitter, heater, antenna, and pad routing."""
    gf.clear_cache()
    component = gf.Component()

    splitter_tree = component << gf.components.splitter_tree(
        noutputs=elements,
        spacing=(50.0, 70.0),
    )
    phase_shifter = gf.components.straight_heater_meander()
    phase_shifter_extended = gf.components.extend_ports(phase_shifter, length=20)

    phase_shifter_optical_ports: list[gf.Port] = []
    phase_shifter_electrical_ports_west: list[gf.Port] = []

    for index, port in enumerate(
        splitter_tree.ports.filter(orientation=0, port_type="optical")
    ):
        ref = component.add_ref(phase_shifter_extended, name=f"ps{index}")
        ref.mirror()
        ref.connect("o1", port)
        component.add_ports(
            ref.ports.filter(port_type="electrical"), prefix=f"ps{index}"
        )
        phase_shifter_optical_ports.append(ref["o2"])
        phase_shifter_electrical_ports_west.append(ref["l_e1"])

    antennas = component << gf.components.array(
        gf.components.dbr(n=200),
        rows=elements,
        columns=1,
        column_pitch=0,
        row_pitch=2.0,
    )
    antennas.xmin = ref.xmax + 50
    antennas.mirror_y()
    antennas.y = 0

    gf.routing.route_bundle(
        component,
        ports1=antennas.ports.filter(orientation=180),
        ports2=phase_shifter_optical_ports,
        radius=5,
        sort_ports=True,
        cross_section="strip",
    )

    pads = component << gf.components.array(
        gf.components.pad,
        rows=len(phase_shifter_electrical_ports_west),
        columns=1,
    )
    pads.xmax = splitter_tree.xmin - 10
    pads.y = 0

    gf.routing.route_bundle_electrical(
        component,
        ports1=pads.ports.filter(orientation=0, port_type="electrical"),
        ports2=phase_shifter_electrical_ports_west,
        sort_ports=True,
        cross_section="metal_routing",
    )
    return component


def build_coherent_tx_dual_pol() -> gf.Component:
    """Nested dual-polarization coherent transmitter sample."""
    gf.clear_cache()
    return coh_tx_dual_pol(combiner="mmi2x2")


def build_coherent_rx_single_pol() -> gf.Component:
    """Single-polarization coherent receiver sample."""
    gf.clear_cache()
    return coh_rx_single_pol()


def build_optical_reticle_pack() -> gf.Component:
    """Packed optical reticle with MZIs, rings, spirals, pads, and IO."""
    gf.clear_cache()
    return sample_reticle(grid=False)


def build_labeled_reticle_pack() -> gf.Component:
    """Packed reticle with optical/electrical test labels."""
    gf.clear_cache()
    return sample_reticle_with_labels(grid=False)


def build_awg_32x8() -> gf.Component:
    """Arrayed waveguide grating with many routed arms."""
    gf.clear_cache()
    return gf.components.awg(
        arms=32,
        outputs=8,
        fpr_spacing=80,
        length_increment=10,
    )


def build_cutback_taper() -> gf.Component:
    """PCM-style daisy chain for cutback loss measurements."""
    gf.clear_cache()
    return gf.components.cutback_component(cols=5, rows=8)


def build_cutback_bend() -> gf.Component:
    """PCM-style bend cutback chain."""
    gf.clear_cache()
    return gf.components.cutback_bend(cols=10, rows=10)


WORKFLOW_FACTORIES: dict[str, ComponentFactory] = {
    "mzi_fiber_array": build_mzi_fiber_array,
    "big_device_fiber_array": build_big_device_fiber_array,
    "big_device_electrical_pads": build_big_device_electrical_pads,
    "lidar_16_with_pads": build_lidar_with_pads,
    "coherent_tx_dual_pol": build_coherent_tx_dual_pol,
    "coherent_rx_single_pol": build_coherent_rx_single_pol,
    "optical_reticle_pack": build_optical_reticle_pack,
    "labeled_reticle_pack": build_labeled_reticle_pack,
    "awg_32x8": build_awg_32x8,
    "cutback_taper_5x8": build_cutback_taper,
    "cutback_bend_10x10": build_cutback_bend,
}

WORKFLOWS = tuple(WORKFLOW_FACTORIES)
ROUNDTRIP_WORKFLOWS = (
    "mzi_fiber_array",
    "lidar_16_with_pads",
    "labeled_reticle_pack",
    "cutback_bend_10x10",
)
NETLIST_WORKFLOWS = (
    "mzi_fiber_array",
    "coherent_rx_single_pol",
    "cutback_taper_5x8",
    "cutback_bend_10x10",
)
RASTER_WORKFLOWS = ("mzi_fiber_array", "awg_32x8")


class TimeEndToEndWorkflowBuild:
    param_names: ClassVar[tuple[str, ...]] = ("workflow",)
    params: ClassVar[tuple[tuple[str, ...], ...]] = (WORKFLOWS,)

    def setup(self, workflow: str) -> None:
        self.factory = WORKFLOW_FACTORIES[workflow]

    def time_build_workflow(self, workflow: str) -> None:
        self.factory()


class TimeEndToEndWorkflowWriteGds:
    param_names: ClassVar[tuple[str, ...]] = ("workflow", "with_metadata")
    params: ClassVar[tuple[tuple[str, ...], tuple[bool, ...]]] = (
        WORKFLOWS,
        (False, True),
    )

    def setup(self, workflow: str, with_metadata: bool) -> None:
        self.component = WORKFLOW_FACTORIES[workflow]()
        self.gds_path = TemporaryGdsPath()

    def teardown(self, workflow: str, with_metadata: bool) -> None:
        self.gds_path.cleanup()

    def time_write_gds(self, workflow: str, with_metadata: bool) -> None:
        self.component.write_gds(
            gdspath=self.gds_path.path,
            with_metadata=with_metadata,
        )


class TimeEndToEndWorkflowImportGds:
    param_names: ClassVar[tuple[str, ...]] = ("workflow",)
    params: ClassVar[tuple[tuple[str, ...], ...]] = (WORKFLOWS,)

    def setup(self, workflow: str) -> None:
        self.gds_path = TemporaryGdsPath()
        component = WORKFLOW_FACTORIES[workflow]()
        component.write_gds(gdspath=self.gds_path.path, with_metadata=True)
        gf.clear_cache()

    def teardown(self, workflow: str) -> None:
        self.gds_path.cleanup()
        gf.clear_cache()

    def time_import_gds(self, workflow: str) -> None:
        gf.clear_cache()
        import_gds(self.gds_path.path)


class TimeEndToEndWorkflowNetlist:
    param_names: ClassVar[tuple[str, ...]] = ("workflow",)
    params: ClassVar[tuple[tuple[str, ...], ...]] = (NETLIST_WORKFLOWS,)

    def setup(self, workflow: str) -> None:
        self.component = WORKFLOW_FACTORIES[workflow]()

    def time_get_netlist_recursive(self, workflow: str) -> None:
        self.component.get_netlist(
            recursive=True,
            on_multi_connect="ignore",
            on_dangling_port="ignore",
        )


class TimeEndToEndWorkflowLabels:
    def setup(self) -> None:
        self.gds_path = TemporaryGdsPath()
        self.csv_path = TemporaryCsvPath()
        self.component = build_labeled_reticle_pack()
        self.component.write_gds(gdspath=self.gds_path.path, with_metadata=False)

    def teardown(self) -> None:
        self.gds_path.cleanup()
        self.csv_path.cleanup()

    def time_write_labels(self) -> None:
        write_labels(
            self.gds_path.path,
            layer_label="TEXT",
            filepath=self.csv_path.path,
            prefixes=("opt-", "elec-"),
        )


class TimeEndToEndWorkflowRaster:
    param_names: ClassVar[tuple[str, ...]] = ("workflow",)
    params: ClassVar[tuple[tuple[str, ...], ...]] = (RASTER_WORKFLOWS,)

    def setup(self, workflow: str) -> None:
        self.component = WORKFLOW_FACTORIES[workflow]()

    def time_to_np(self, workflow: str) -> None:
        to_np(
            self.component,
            nm_per_pixel=500,
            layers=((1, 0),),
            pad_width=0,
        )


class TimeEndToEndWorkflowRoundTrip:
    param_names: ClassVar[tuple[str, ...]] = ("workflow",)
    params: ClassVar[tuple[tuple[str, ...], ...]] = (ROUNDTRIP_WORKFLOWS,)

    def setup(self, workflow: str) -> None:
        self.factory = WORKFLOW_FACTORIES[workflow]
        self.gds_path = TemporaryGdsPath()

    def teardown(self, workflow: str) -> None:
        self.gds_path.cleanup()
        gf.clear_cache()

    def time_build_write_import_gds(self, workflow: str) -> None:
        component = self.factory()
        component.write_gds(gdspath=self.gds_path.path, with_metadata=True)
        gf.clear_cache()
        import_gds(self.gds_path.path)
