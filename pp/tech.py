import pathlib
from dataclasses import field
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import pydantic.dataclasses as dataclasses
from phidl.device_layout import Device as Component

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
    zmin_nm: Optional[float] = None
    material: Optional[str] = None


@dataclasses.dataclass
class LayerStack:
    WG = LayerLevel((1, 0), thickness_nm=220.0, zmin_nm=0.0, material="si")
    WGCLAD = LayerLevel((111, 0), zmin_nm=0.0, material="sio2")
    SLAB150 = LayerLevel((2, 0), thickness_nm=150.0, zmin_nm=0, material="si")
    SLAB90 = LayerLevel((3, 0), thickness_nm=150.0, zmin_nm=0.0, material="si")
    WGN = LayerLevel((34, 0), thickness_nm=350.0, zmin_nm=220.0 + 100.0, material="sin")
    WGN_CLAD = LayerLevel((36, 0))

    def get_layer_to_thickness_nm(self) -> Dict[Tuple[int, int], float]:
        """Returns layer tuple to thickness_nm."""
        return {
            getattr(self, key).layer: getattr(self, key).thickness_nm
            for key in dir(self)
            if not key.startswith(IGNORE_PREXIXES) and getattr(self, key).thickness_nm
        }

    def get_layer_to_zmin_nm(self) -> Dict[Tuple[int, int], float]:
        """Returns layer tuple to z min position (nm)."""
        return {
            getattr(self, key).layer: getattr(self, key).zmin_nm
            for key in dir(self)
            if not key.startswith(IGNORE_PREXIXES)
            and getattr(self, key).zmin_nm is not None
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


# waveguides


@dataclasses.dataclass
class Section:
    width: float
    offset: float = 0
    layer: Layer = (1, 0)
    ports: Tuple[Optional[str], Optional[str]] = (None, None)
    name: str = None


@dataclasses.dataclass
class Waveguide:
    width: float
    layer: Layer
    width_wide: Optional[float] = None
    auto_widen: bool = False
    auto_widen_minimum_length: float = 200
    taper_length: float = 10.0
    radius: float = 10.0
    cladding_offset: Optional[float] = 3.0
    layer_cladding: Optional[Layer] = None
    layers_cladding: Optional[List[Layer]] = None
    sections: Optional[Tuple[Section, ...]] = None


@dataclasses.dataclass
class Strip(Waveguide):
    width: float = 0.5
    width_wide: float = 2.0
    auto_widen: bool = True
    auto_widen_minimum_length: float = 200
    taper_length: float = 10.0
    layer: Layer = LAYER.WG
    radius: float = 10.0
    cladding_offset: float = 3.0
    layer_cladding: Optional[Layer] = LAYER.WGCLAD
    layers_cladding: Optional[List[Layer]] = (LAYER.WGCLAD,)


@dataclasses.dataclass
class Rib(Waveguide):
    width: float = 0.5
    auto_widen: bool = True
    auto_widen_minimum_length: float = 200
    taper_length: float = 10.0
    layer: Layer = LAYER.WG
    radius: float = 10.0
    cladding_offset: float = 3.0
    layer_cladding: Optional[Layer] = LAYER.SLAB90
    layers_cladding: Optional[List[Layer]] = (LAYER.SLAB90,)


@dataclasses.dataclass
class MetalRouting(Waveguide):
    width: float = 2.0
    width_wide: float = 2.0
    auto_widen: bool = False
    layer: Layer = LAYER.M3
    radius: float = 5.0


@dataclasses.dataclass
class Nitride(Waveguide):
    width: float = 1.0
    width_wide: float = 1.0
    auto_widen: bool = False
    layer: Layer = LAYER.WGN
    radius: float = 20.0


@dataclasses.dataclass
class StripHeater(Waveguide):
    width: float = 1.0
    auto_widen: bool = False
    layer: Layer = LAYER.WG
    radius: float = 10.0
    sections: Tuple[Section, ...] = (
        Section(width=8, layer=LAYER.WGCLAD),
        Section(
            width=0.5, layer=LAYER.HEATER, offset=+1.2, ports=("top_in", "top_out")
        ),
        Section(
            width=0.5, layer=LAYER.HEATER, offset=-1.2, ports=("bot_in", "bot_out")
        ),
    )


@dataclasses.dataclass
class Waveguides:
    strip: Waveguide = Strip()
    metal_routing: Waveguide = MetalRouting()
    nitride: Waveguide = Nitride()
    strip_heater: Waveguide = StripHeater()
    rib: Waveguide = StripHeater()


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


# Component settings


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
    with_cladding_box: bool = True
    waveguide: str = "strip"


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
    layer: Layer = LAYER.M3
    size: float = 10


@dataclasses.dataclass
class ComponentSettings:
    pad: Pad = Pad()
    mmi1x2: Mmi1x2 = Mmi1x2()
    mmi2x2: Mmi2x2 = Mmi2x2()
    manhattan_text: ManhattanText = ManhattanText()


def make_empty_dict():
    return {}


@dataclasses.dataclass
class Factory:
    """Stores component factories

    Args:
        factory: component name to function
        settings: Optional component settings with defaults
        add_pins: Optional function to add pins
    """

    factory: Dict[str, Callable] = field(default_factory=make_empty_dict)
    settings: ComponentSettings = ComponentSettings()
    add_pins: Optional[Callable[[], None]] = None

    def register(self, factory: Union[Iterable[Callable], Callable]) -> None:
        """Registers component_factory into the factory."""

        if hasattr(factory, "__iter__"):
            for i in factory:
                self.register(i)
            return
        if not callable(factory):
            raise ValueError(
                f"Error: expected callable: got factory = {factory} with type {type(factory)}"
            )

        self.factory[factory.__name__] = factory
        # setattr(self, factory.__name__, factory)

    def get_component(
        self,
        component_type: str,
        **settings,
    ) -> Component:
        """Returns Component from factory.
        Takes default settings from self.settings
        settings can be overwriten with kwargs

        Args:
            component_type:
            **settings
        """
        if component_type not in self.factory:
            raise ValueError(f"{component_type} not in {list(self.factory.keys())}")
        component_settings = getattr(self.settings, component_type, {})
        component_settings.update(**settings)
        component = self.factory[component_type](**component_settings)
        if self.add_pins:
            self.add_pins(component)
        return component


# TECH
@dataclasses.dataclass
class Tech:
    name: str = "generic"
    layer: LayerMap = LAYER

    fiber_spacing: float = 50.0
    fiber_array_spacing: float = 127.0
    fiber_input_to_output_spacing: float = 200.0
    layer_label: Layer = LAYER.LABEL
    metal_spacing: float = 10.0

    snap_to_grid_nm: int = 1
    layer_stack: LayerStack = LAYER_STACK
    waveguide: Waveguides = WAVEGUIDES

    sparameters_path: str = str(module_path / "gdslib" / "sparameters")
    simulation_settings: SimulationSettings = SIMULATION_SETTINGS
    component_settings: ComponentSettings = ComponentSettings()


TECH = Tech()

if __name__ == "__main__":
    import pp

    # t = TECH
    # c = pp.c.mmi1x2(length_mmi=25.5)
    # t.register_component(c)

    def mmi1x2_longer(length_mmi: float = 25.0, **kwargs):
        return pp.c.mmi1x2(length_mmi=length_mmi, **kwargs)

    def mzi_longer(**kwargs):
        return pp.c.mzi(splitter=mmi1x2_longer, **kwargs)

    # t.register_component_factory(mmi1x2_longer)
    # c = t.component.mmi1x2_longer(length_mmi=30)
    # c.show()

    cf = Factory()
    cf.register(mmi1x2_longer)
    # cf.register(mmi1x2_longer())
    c = cf.get_component("mmi1x2_longer", length_mmi=30)
    c.show()
