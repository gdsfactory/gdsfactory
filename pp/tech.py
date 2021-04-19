import io
import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple

import omegaconf

from pp.config import CONFIG
from pp.layers import LAYER, LAYER_STACK, LayerStack
from pp.types import Layer

component_settings = omegaconf.OmegaConf.load(
    io.StringIO(
        """
tech:
    wg_width: 0.5
    bend_radius: 5.0
    cladding_offset: 3.0
    layer: [1, 0]
    layer_heater: [47, 0]
    layer_label: [201, 0]


mmi1x2:
    width: 0.5
    width_taper: 1.0
    length_taper: 10.0
    length_mmi: 5.5
    width_mmi: 2.5
    gap_mmi: 0.25

"""
    )
)


@dataclass
class SimulationSettings:
    remove_layers: Tuple[Layer, ...] = (LAYER.WGCLAD,)
    background_material: str = "sio2"
    port_width: float = 3e-6
    port_height: float = 1.5e-6
    port_extension_um: float = 1.0
    mesh_accuracy: int = 2
    zmargin: float = 1e-6
    ymargin: float = 2e-6
    wavelength_start: float = 1.2e-6
    wavelength_stop: float = 1.6e-6
    wavelength_points: int = 500


simulation_settings = SimulationSettings()


@dataclass(frozen=True)
class Tech:
    name: str
    wg_width: float
    bend_radius: float
    cladding_offset: float
    layer_wg: Layer
    layers_cladding: Optional[Tuple[Layer, ...]]
    layer_label: Layer
    taper_length: float
    taper_width: float
    fiber_single_spacing: float = 50.0
    fiber_array_spacing: float = 127.0
    fiber_input_to_output_spacing: float = 200.0
    snap_to_grid_nm: int = 1
    auto_widen: bool = False
    sparameters_path: pathlib.Path = CONFIG["sp"]
    simulation_settings: SimulationSettings = simulation_settings
    layer_stack: LayerStack = LAYER_STACK
    component_settings: omegaconf.dictconfig.DictConfig = component_settings


@dataclass(frozen=True)
class TechSiliconCband(Tech):
    name: str = "silicon_cband"
    wg_width: float = 0.5
    bend_radius: float = 10.0
    cladding_offset: float = 3.0
    layer_wg: Layer = LAYER.WG
    layers_cladding: Tuple[Layer, ...] = (LAYER.WGCLAD,)
    layer_label: Layer = LAYER.LABEL
    layer_heater: Layer = LAYER.HEATER
    taper_length: float = 15.0
    taper_width: float = 2.0  # taper to wider straights for lower loss


@dataclass(frozen=True)
class TechNitrideCband(Tech):
    name: str = "nitride_cband"
    wg_width: float = 1.0
    bend_radius: float = 20.0
    cladding_offset: float = 3.0
    layer_wg: Layer = LAYER.WGN
    layers_cladding: Tuple[Layer, ...] = (LAYER.WGN_CLAD,)
    layer_label: Layer = LAYER.LABEL
    taper_length: float = 20.0
    taper_width: float = 1.0


@dataclass(frozen=True)
class TechMetal1(Tech):
    name: str = "metal1"
    wg_width: float = 2.0
    bend_radius: float = 10.0
    cladding_offset: float = 3.0
    layer_wg: Layer = LAYER.M1
    layers_cladding: Tuple[Layer, ...] = (LAYER.WGCLAD,)
    layer_label: Layer = LAYER.LABEL
    taper_length: float = 20.0
    taper_width: float = 10.0
    snap_to_grid_nm: int = 10


TECH_SILICON_C = TechSiliconCband()
TECH_NITRIDE_C = TechNitrideCband()
TECH_METAL1 = TechMetal1()


if __name__ == "__main__":
    import pp

    c = pp.components.straight(tech=TECH_METAL1)
    print(c.name)

    from dataclasses import asdict

    tech = TECH_METAL1
    print(asdict(tech.simulation_settings))
