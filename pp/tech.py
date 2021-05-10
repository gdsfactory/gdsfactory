import dataclasses
import pathlib
from typing import Dict, Optional, Tuple

module_path = pathlib.Path(__file__).parent.absolute()
Layer = Tuple[int, int]
IGNORE_PREXIXES = ("_", "get_")


@dataclasses.dataclass
class LayerMap:
    WG: Layer = (1, 0)
    WGCLAD: Layer = (111, 0)
    SLAB150: Layer = (2, 0)
    SLAB90: Layer = (3, 0)
    WGN: Layer = (34, 0)
    WGN_CLAD: Layer = (36, 0)
    N: Layer = (20, 0)
    Np: Layer = (22, 0)
    Npp: Layer = (24, 0)
    P: Layer = (21, 0)
    Pp: Layer = (23, 0)
    Ppp: Layer = (25, 0)
    HEATER: Layer = (47, 0)
    M1: Layer = (41, 0)
    M2: Layer = (45, 0)
    M3: Layer = (49, 0)
    VIA1: Layer = (40, 0)
    VIA2: Layer = (44, 0)
    VIA3: Layer = (43, 0)
    NO_TILE_SI: Layer = (71, 0)
    DEEPTRENCH: Layer = (7, 0)
    PADDING: Layer = (67, 0)
    DEVREC: Layer = (68, 0)
    FLOORPLAN: Layer = (64, 0)
    TEXT: Layer = (66, 0)
    PORT: Layer = (1, 10)
    PORTE: Layer = (69, 0)
    PORTH: Layer = (70, 0)
    LABEL: Layer = (201, 0)
    LABEL_SETTINGS: Layer = (202, 0)
    TE: Layer = (203, 0)
    TM: Layer = (204, 0)
    DRC_MARKER: Layer = (205, 0)
    LABEL_INSTANCE: Layer = (206, 0)


LAYER = LayerMap()


@dataclasses.dataclass
class LayerLevel:
    """Layer For 3D LayerStack.

    Args:
        layer: GDSII Layer
        thickness_nm: thickness of layer
        z_nm: height position where material starts
        material: material name
    """

    layer: Tuple[int, int]
    thickness_nm: Optional[float] = None
    z_nm: Optional[float] = None
    material: Optional[str] = None


@dataclasses.dataclass
class LayerStack:
    WG = LayerLevel((1, 0), thickness_nm=220.0, z_nm=0.0, material="si")
    WGCLAD = LayerLevel((111, 0), z_nm=0.0, material="sio2")
    SLAB150 = LayerLevel((2, 0), thickness_nm=150.0, z_nm=0, material="si")
    SLAB90 = LayerLevel((3, 0), thickness_nm=150.0, z_nm=0.0, material="si")
    WGN = LayerLevel((34, 0), thickness_nm=350.0, z_nm=220.0 + 100.0, material="sin")
    WGN_CLAD = LayerLevel((36, 0))

    def get_layer_to_thickness_nm(self) -> Dict[Tuple[int, int], float]:
        """Returns layer tuple to thickness_nm."""
        return {
            getattr(self, key).layer: getattr(self, key).thickness_nm
            for key in dir(self)
            if not key.startswith(IGNORE_PREXIXES) and getattr(self, key).thickness_nm
        }

    def get_layer_to_material(self) -> Dict[Tuple[int, int], float]:
        """Returns layer tuple to material."""
        return {
            getattr(self, key).layer: getattr(self, key).material
            for key in dir(self)
            if not key.startswith(IGNORE_PREXIXES) and getattr(self, key).material
        }

    def get_from_tuple(self, layer_tuple: Tuple[int, int]) -> str:
        """Returns Layer from layer tuple (gds_layer, gds_datatype)."""
        tuple_to_name = {
            getattr(self, name).layer: name
            for name in dir(self)
            if not name.startswith(IGNORE_PREXIXES)
        }
        if layer_tuple not in tuple_to_name:
            raise ValueError(f"Layer {layer_tuple} not in {list(tuple_to_name.keys())}")

        name = tuple_to_name[layer_tuple]
        return name


LAYER_STACK = LayerStack()


@dataclasses.dataclass
class Strip:
    width: float = 0.5
    width_wide: float = 2.0
    auto_widen: bool = True
    auto_widen_minimum_length: float = 200
    taper_length: float = 10.0
    layer: Layer = LAYER.WG
    radius: float = 10.0
    cladding_offset: float = 3.0
    layer_cladding: Optional[Layer] = LAYER.WGCLAD
    layers_cladding: Optional[Tuple[Layer]] = (LAYER.WGCLAD,)


@dataclasses.dataclass
class MetalRouting:
    width: float = 2.0
    width_wide: float = 2.0
    auto_widen: bool = False
    layer: Layer = LAYER.M3
    radius: float = 10.0
    min_spacing: float = 10.0


@dataclasses.dataclass
class Nitride:
    width: float = 1.0
    width_wide: float = 1.0
    auto_widen: bool = False
    layer: Layer = LAYER.WGN
    radius: float = 20.0


STRIP = Strip()
METAL_ROUTING = MetalRouting()


@dataclasses.dataclass
class Waveguides:
    strip: Strip = Strip()
    metal_routing: MetalRouting = MetalRouting()
    nitride: Nitride = Nitride()
    # rib: Rib
    # strip_heater: StripHeater


@dataclasses.dataclass
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


WAVEGUIDES = Waveguides()
SIMULATION_SETTINGS = SimulationSettings()


@dataclasses.dataclass
class Pad:
    width: float = 100
    height: float = 100
    layer: Layer = LAYER.M3


@dataclasses.dataclass
class Mmi1x2:
    width: float = 0.5
    width_taper: float = 1.0
    length_taper: float = 10.0
    length_mmi: float = 5.5
    width_mmi: float = 2.5
    gap_mmi: float = 0.25


@dataclasses.dataclass
class Mmi2x2:
    width: float = 0.5
    width_taper: float = 1.0
    length_taper: float = 10.0
    length_mmi: float = 5.5
    width_mmi: float = 2.5
    gap_mmi: float = 0.25


@dataclasses.dataclass
class ManhattanText:
    layer: str = LAYER.M3
    size: float = 10


@dataclasses.dataclass
class Components:
    pad: Pad = Pad()
    mmi1x2: Mmi1x2 = Mmi1x2()
    mmi2x2: Mmi2x2 = Mmi2x2()
    manhattan_text: ManhattanText = ManhattanText()


@dataclasses.dataclass
class Tech:
    name: str = "generic"
    layer: LayerMap = LAYER

    fiber_spacing: float = 50.0
    fiber_array_spacing: float = 127.0
    fiber_input_to_output_spacing: float = 200.0
    layer_label: Layer = LAYER.LABEL

    snap_to_grid_nm: int = 1
    layer_stack: LayerStack = LAYER_STACK
    waveguide: Waveguides = WAVEGUIDES
    components: Components = Components()

    sparameters_path: str = str(module_path / "gdslibg" / "sparameters")
    simulation_settings: SimulationSettings = SIMULATION_SETTINGS


TECH = Tech()

if __name__ == "__main__":
    pass
