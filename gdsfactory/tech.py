import pathlib
from dataclasses import asdict, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import pydantic
from phidl.device_layout import Device as Component

from gdsfactory.layers import LayerSet
from gdsfactory.name import clean_value, get_name_short

module_path = pathlib.Path(__file__).parent.absolute()
Layer = Tuple[int, int]


@pydantic.dataclasses.dataclass(frozen=True)
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
    FLOORPLAN: Layer = (99, 0)
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
PORT_MARKER_LAYER_TO_TYPE = {
    LAYER.PORT: "optical",
    LAYER.PORTE: "dc",
    LAYER.TE: "vertical_te",
    LAYER.TM: "vertical_tm",
}

PORT_LAYER_TO_TYPE = {
    LAYER.WG: "optical",
    LAYER.WGN: "optical",
    LAYER.SLAB150: "optical",
    LAYER.M1: "dc",
    LAYER.M2: "dc",
    LAYER.M3: "dc",
    LAYER.TE: "vertical_te",
    LAYER.TM: "vertical_tm",
}

PORT_TYPE_TO_MARKER_LAYER = {v: k for k, v in PORT_MARKER_LAYER_TO_TYPE.items()}
LAYER_SET = LayerSet()  # Layerset for simulation and matplotlib


@pydantic.dataclasses.dataclass
class LayerLevel:
    """Layer For 3D LayerStack.

    Args:
        name: Name of the Layer.
        gds_layer: GDSII Layer number.
        gds_datatype: GDSII datatype.
        thickness_nm: thickness of layer
        zmin_nm: height position where material starts
        material: material name
        sidewall_angle: in degrees with respect to normal
    """

    name: str
    gds_layer: int
    gds_datatype: int = 0
    thickness_nm: Optional[float] = None
    zmin_nm: Optional[float] = None
    material: Optional[str] = None
    sidewall_angle: float = 0

    @property
    def gds(self):
        return (self.gds_layer, self.gds_datatype)


@pydantic.dataclasses.dataclass
class LayerStack:
    """
    For simulation and matplotlib

    """

    layers: List[LayerLevel]

    def get_layer_to_thickness_nm(self) -> Dict[Tuple[int, int], float]:
        """Returns layer tuple to thickness_nm."""
        return {
            layer.gds: layer.thickness_nm for layer in self.layers if layer.thickness_nm
        }

    def get_layer_to_zmin_nm(self) -> Dict[Tuple[int, int], float]:
        """Returns layer tuple to z min position (nm)."""
        return {layer.gds: layer.zmin_nm for layer in self.layers if layer.thickness_nm}

    def get_layer_to_material(self) -> Dict[Tuple[int, int], float]:
        """Returns layer tuple to material."""
        return {
            layer.gds: layer.material for layer in self.layers if layer.thickness_nm
        }

    def to_dict(self):
        return {layer.name: asdict(layer) for layer in self.layers}


def get_layer_stack_generic(thickness_nm: float = 220.0) -> LayerStack:
    """Returns generic LayerStack"""
    return LayerStack(
        layers=[
            LayerLevel(
                name="core",
                gds_layer=1,
                thickness_nm=thickness_nm,
                zmin_nm=0.0,
                material="si",
            ),
            LayerLevel(
                name="clad",
                gds_layer=111,
                zmin_nm=0.0,
                material="sio2",
            ),
            LayerLevel(
                name="slab150",
                gds_layer=2,
                thickness_nm=150.0,
                zmin_nm=0,
                material="si",
            ),
            LayerLevel(
                name="slab90",
                gds_layer=3,
                thickness_nm=150.0,
                zmin_nm=0.0,
                material="si",
            ),
            LayerLevel(
                name="nitride",
                gds_layer=34,
                thickness_nm=350.0,
                zmin_nm=220.0 + 100.0,
                material="sin",
            ),
            LayerLevel(name="nitride_clad", gds_layer=36),
        ]
    )


LAYER_STACK = get_layer_stack_generic()


@pydantic.dataclasses.dataclass
class Section:
    """
    Args:
        width: of the section
        offset: center to center
        layer:
        ports: optional name of the ports
        name: optional section name
    """

    width: float
    offset: float = 0
    layer: Layer = (1, 0)
    ports: Tuple[Optional[str], Optional[str]] = (None, None)
    name: Optional[str] = None
    port_types: Tuple[str, str] = ("optical", "optical")

    def __repr__(self):
        return "_".join(
            [
                f"{i}"
                for i in [
                    self.name,
                    int(self.width * 1e3),
                    self.layer[0],
                    self.layer[1],
                    self.ports[0],
                    self.ports[1],
                    self.port_types[0],
                    self.port_types[1],
                ]
                if i is not None
            ]
        )


@pydantic.dataclasses.dataclass
class SimulationSettings:
    background_material: str = "sio2"
    port_width: float = 3e-6
    port_height: float = 1.5e-6
    port_extension_um: float = 2.0
    mesh_accuracy: int = 2
    zmargin: float = 1e-6
    ymargin: float = 2e-6
    xmargin: float = 0.5e-6
    pml_margin: float = 0.5e-6
    wavelength_start: float = 1.2e-6
    wavelength_stop: float = 1.6e-6
    wavelength_points: int = 500


SIMULATION_SETTINGS = SimulationSettings()


def make_empty_dict():
    return {}


def assert_callable(function):
    if not callable(function):
        raise ValueError(
            f"Error: expected callable: got function = {function} with type {type(function)}"
        )


@pydantic.dataclasses.dataclass
class Library:
    """Stores component factories for defining a factory of Components.

    Args:
        factory: component name to function
        settings: Optional component settings with defaults
        post_init: Optional function to run over component

    """

    name: str
    factory: Dict[str, Callable] = field(default_factory=make_empty_dict)
    post_init: Optional[Callable[[], None]] = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_component_names(self):
        return list(self.factory.keys())

    def register(
        self,
        function_or_function_list: Optional[
            Union[List[Callable[[], None]], Callable[[], None]]
        ] = None,
        **kwargs,
    ) -> None:
        """Registers component_factory function or functions into the factory."""
        function_or_function_list = function_or_function_list or []

        if not hasattr(function_or_function_list, "__iter__"):
            function = function_or_function_list
            assert_callable(function)
            self.factory[function.__name__] = function

        else:
            for function in function_or_function_list:
                assert_callable(function)
                self.factory[get_name_short(clean_value(function))] = function

        for function_name, function in kwargs.items():
            assert_callable(function)
            self.factory[function_name] = function

    def get_component(
        self,
        component: Union[str, Dict],
        **settings,
    ) -> Component:
        """Returns Component from library.
        Takes default settings from self.settings
        settings can be overwriten with kwargs
        runs any post_init functions over the component before it returns it.

        Priority (from lower to higher)
        - default in dataclass
        - component in case it's a dic
        - **settings

        Args:
            component:
            **settings
        """
        if isinstance(component, str):
            component = component
            component_settings = {}
        elif isinstance(component, dict):
            component_settings = component.copy()
            if "component" not in component_settings:
                raise ValueError(f"{component} is missing `component` key")

            component = component_settings.pop("component")
        else:
            raise ValueError(
                f"{component} needs to be a string or dict, got {type(component)}"
            )

        if component not in self.factory:
            raise ValueError(f"{component} not in {list(self.factory.keys())}")
        component = self.factory[component](**settings)
        if self.post_init and not hasattr(component, "_initialized"):
            self.post_init(component)
            component._initialized = True
        return component


@pydantic.dataclasses.dataclass
class Tech:
    name: str = "generic"
    layer: LayerMap = LAYER

    fiber_spacing: float = 50.0
    fiber_array_spacing: float = 127.0
    fiber_input_to_output_spacing: float = 200.0
    layer_label: Layer = LAYER.LABEL
    metal_spacing: float = 10.0

    layer_stack: LayerStack = LAYER_STACK
    sparameters_path: str = str(module_path / "gdslib" / "sparameters")
    simulation_settings: SimulationSettings = SIMULATION_SETTINGS


TECH = Tech()
LIBRARY = Library("generic_components")

if __name__ == "__main__":
    import gdsfactory as gf

    def mmi1x2_longer(length_mmi: float = 25.0, **kwargs):
        return gf.components.mmi1x2(length_mmi=length_mmi, **kwargs)

    def mzi_longer(**kwargs):
        return gf.components.mzi(splitter=mmi1x2_longer, **kwargs)

    # ls = LAYER_STACK
    # print(ls.get_layer_to_material())
    # print(ls.get_layer_to_thickness_nm())

    s = Section(width=1, layer=(1, 0))
    print(s)
