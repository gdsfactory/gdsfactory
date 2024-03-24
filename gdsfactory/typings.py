"""In programming, a factory is a function that returns an object.

Functions are easy to understand because they have clear inputs and outputs.
Most gdsfactory functions take some inputs and return a Component object.
Some of these inputs parameters are also functions.

- Component: Object with.
    - name.
    - references: to other components (x, y, rotation).
    - polygons in different layers.
    - ports dict.
- Route: dataclass with 3 attributes.
    - references: list of references (straights, bends and tapers).
    - ports: dict(input=PortIn, output=PortOut).
    - length: how long is this route?

Factories:

- ComponentFactory: function that returns a Component.
- RouteFactory: function that returns a Route.


Specs:

- ComponentSpec: Component, function, string or dict
    (component=mzi, settings=dict(delta_length=20)).
- LayerSpec: (3, 0), 3 (assumes 0 as datatype) or string.

"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Callable
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import gdstk
import numpy as np
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, root_validator

from gdsfactory.component import Component, ComponentReference
from gdsfactory.component_layout import Label
from gdsfactory.cross_section import CrossSection, Section, Transition, WidthTypes
from gdsfactory.port import Port
from gdsfactory.technology import LayerLevel, LayerMap, LayerStack

STEP_DIRECTIVES = {
    "x",
    "y",
    "dx",
    "dy",
}

STEP_DIRECTIVES_ALL_ANGLE = {
    "x",
    "y",
    "dx",
    "dy",
    "ds",
    "exit_angle",
    "cross_section",
    "connector",
    "separation",
}


@dataclasses.dataclass
class Step:
    """Manhattan Step.

    Parameters:
        x: absolute.
        y: absolute.
        dx: x-displacement.
        dy: y-displacement.

    """

    x: float | None = None
    y: float | None = None
    dx: float | None = None
    dy: float | None = None


@dataclasses.dataclass
class StepAllAngle:
    x: float | None = None
    y: float | None = None
    dx: float | None = None
    dy: float | None = None
    ds: float | None = None
    exit_angle: float | None = None
    cross_section: CrossSectionSpec | None = None
    connector: ComponentSpec | None = None
    separation: float | None = None

    """All angle Ste.

    Parameters:
        x: absolute.
        y: absolute.
        dx: x-displacement.
        dy: y-displacement.
        exit_angle: in degrees.
        cross_section: spec.
        connector: define transition.
        separation: in um.

    """
    model_config = {"extra": "forbid", "frozen": True}


Anchor = Literal[
    "ce",
    "cw",
    "nc",
    "ne",
    "nw",
    "sc",
    "se",
    "sw",
    "center",
    "cc",
]
AnchorSubset = Literal[
    "center",
    "l",
    "r",
    "s",
]
Axis = Literal["x", "y"]
NSEW = Literal["N", "S", "E", "W"]


Float2 = tuple[float, float]
Float3 = tuple[float, float, float]
Floats = tuple[float, ...]
Strs = tuple[str, ...]
Int2 = tuple[int, int]
Int3 = tuple[int, int, int]
Ints = tuple[int, ...]

Layer = tuple[int, int]  # Tuple of integer (layer, datatype)
Layers = tuple[Layer, ...]
LayerSpec = Layer | str  # tuple of integers (layer, datatype) or a string (layer_name)

LayerSpecs = list[LayerSpec] | tuple[LayerSpec, ...] | set[LayerSpec]
ComponentFactory = Callable[..., Component]
ComponentFactoryDict = dict[str, ComponentFactory]
PathType = str | pathlib.Path
PathTypes = tuple[PathType, ...]


MaterialSpec = str | float | tuple[float, float] | Callable

ComponentOrPath = PathType | Component
ComponentOrReference = Component | ComponentReference
NameToFunctionDict = dict[str, ComponentFactory]
Number = float | int
Coordinate = tuple[float, float]
Coordinates = tuple[Coordinate, ...] | list[Coordinate]
ComponentOrPath = Component | PathType
CrossSectionFactory = Callable[..., CrossSection]
TransitionFactory = Callable[..., Transition]
CrossSectionOrFactory = CrossSection | Callable[..., CrossSection]
PortSymmetries = dict[str, list[str]]
PortsDict = dict[str, Port]
PortsList = dict[str, Port]
Metadata = dict[str, int | float | str]

Sparameters = dict[str, np.ndarray]

ComponentSpec = (
    str | ComponentFactory | Component | dict[str, Any]
)  # PCell function, function name, dict or Component

ComponentSpecs = list[ComponentSpec] | tuple[ComponentSpec, ...]
ComponentSpecOrList = ComponentSpec | ComponentSpecs
CellSpec = (
    str | ComponentFactory | dict[str, Any]
)  # PCell function, function name or dict

ComponentSpecDict = dict[str, ComponentSpec]
CrossSectionSpec = (
    CrossSectionFactory | CrossSection | dict[str, Any] | str | Transition
)
CrossSectionSpecs = tuple[CrossSectionSpec, ...]

MultiCrossSectionAngleSpec = list[tuple[CrossSectionSpec, tuple[int, ...]]]

LabelListFactory = Callable[..., list[Label]]

ConductorConductorName = tuple[str, str]
ConductorViaConductorName = tuple[str, str, str] | tuple[str, str]
ConnectivitySpec = ConductorConductorName | ConductorViaConductorName


class Route(BaseModel):
    references: list[ComponentReference]
    labels: list[gdstk.Label] | None = None
    ports: tuple[Port, Port]
    length: float

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}


class Routes(BaseModel):
    references: list[ComponentReference]
    lengths: list[float]
    ports: list[Port] | None = None
    bend_radius: list[float] | None = None

    model_config = {"extra": "forbid"}


class Instance(BaseModel):
    component: str
    settings: dict[str, Any] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict, exclude=True)

    model_config = {"extra": "forbid"}

    @root_validator(pre=True)
    def update_settings_and_info(cls, values):
        """Validator to update component, settings and info based on the component."""
        component = values.get("component")
        settings = values.get("settings", {})
        info = values.get("info", {})

        import gdsfactory as gf

        c = gf.get_component(component)
        component_info = c.info.model_dump(exclude_none=True)
        component_settings = c.settings.model_dump(exclude_none=True)
        values["info"] = {**component_info, **info}
        values["settings"] = {**component_settings, **settings}
        values["component"] = c.function_name
        return values


class Placement(BaseModel):
    x: str | float = 0
    y: str | float = 0
    xmin: str | float | None = None
    ymin: str | float | None = None
    xmax: str | float | None = None
    ymax: str | float | None = None
    dx: float = 0
    dy: float = 0
    port: str | Anchor | None = None
    rotation: int = 0
    mirror: bool = False

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key, 0)

    model_config = {"extra": "forbid"}


class Bundle(BaseModel):
    links: dict[str, str]
    settings: dict[str, Any] = Field(default_factory=dict)
    routing_strategy: str = "get_bundle"

    model_config = {"extra": "forbid"}


class Netlist(BaseModel):
    """Netlist defined component.

    Parameters:
        instances: dict of instances (name, settings, component).
        placements: dict of placements.
        connections: dict of connections.
        routes: dict of routes.
        name: component name.
        info: information (polarization, wavelength ...).
        ports: exposed component ports.
        settings: input variables.
    """

    instances: dict[str, Instance] = Field(default_factory=dict)
    placements: dict[str, Placement] = Field(default_factory=dict)
    connections: dict[str, str] = Field(default_factory=dict)
    routes: dict[str, Bundle] = Field(default_factory=dict)
    name: str | None = None
    info: dict[str, Any] = Field(default_factory=dict)
    ports: dict[str, str] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict, exclude=True)

    model_config = {"extra": "forbid"}


_route_counter = 0


class Net(BaseModel):
    """Net between two ports.

    Parameters:
        ip1: instance_name,port 1.
        ip2: instance_name,port 2.
        name: route name.
    """

    ip1: str
    ip2: str
    settings: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None

    def __init__(self, **data):
        global _route_counter
        super().__init__(**data)
        # If route name is not provided, generate one automatically
        if self.name is None:
            self.name = f"route_{_route_counter}"
            _route_counter += 1


class Schematic(BaseModel):
    """Schematic."""

    netlist: Netlist = Field(default_factory=Netlist)
    nets: list[Net] = Field(default_factory=list)
    placements: dict[str, Placement] = Field(default_factory=dict)

    def add_instance(
        self, name: str, instance: Instance, placement: Placement | None = None
    ) -> None:
        self.netlist.instances[name] = instance
        if placement:
            self.add_placement(name, placement)

    def add_placement(
        self,
        instance_name: str,
        placement: Placement,
    ) -> None:
        """Add placement to the netlist.

        Args:
            instance_name: instance name.
            placement: placement.
        """
        self.placements[instance_name] = placement
        self.netlist.placements[instance_name] = placement

    def from_component(self, component: Component) -> None:
        n = component.get_netlist()
        self.netlist = Netlist.model_validate(n)

    def add_net(self, net: Net) -> None:
        """Add a net between two ports."""
        self.nets.append(net)
        if net.name not in self.netlist.routes:
            self.netlist.routes[net.name] = Bundle(
                links={net.ip1: net.ip2}, settings=net.settings
            )
        else:
            self.netlist.routes[net.name].links[net.ip1] = net.ip2

    def plot_netlist(
        self,
        with_labels: bool = True,
        font_weight: str = "normal",
    ):
        """Plots a netlist graph with networkx.

        Args:
            with_labels: add label to each node.
            font_weight: normal, bold.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        plt.figure()
        netlist = self.netlist
        connections = netlist.connections
        placements = self.placements if self.placements else netlist.placements
        G = nx.Graph()
        G.add_edges_from(
            [
                (",".join(k.split(",")[:-1]), ",".join(v.split(",")[:-1]))
                for k, v in connections.items()
            ]
        )
        pos = {k: (v["x"], v["y"]) for k, v in placements.items()}
        labels = {k: ",".join(k.split(",")[:1]) for k in placements.keys()}

        for node, placement in placements.items():
            if not G.has_node(
                node
            ):  # Check if the node is already in the graph (from connections), to avoid duplication.
                G.add_node(node)
                pos[node] = (placement.x, placement.y)

        for net in self.nets:
            G.add_edge(net.ip1.split(",")[0], net.ip2.split(",")[0])

        nx.draw(
            G,
            with_labels=with_labels,
            font_weight=font_weight,
            labels=labels,
            pos=pos,
        )
        return G


RouteFactory = Callable[..., Route]


class TypedArray(np.ndarray):
    """based on https://github.com/samuelcolvin/pydantic/issues/380."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val, _info):
        return np.array(val, dtype=cls.inner_type)


class ArrayMeta(type):
    def __getitem__(self, t):
        return type("Array", (TypedArray,), {"inner_type": t})


class Array(np.ndarray, metaclass=ArrayMeta):
    pass


__all__ = (
    "Any",
    "Callable",
    "Component",
    "ComponentFactory",
    "ComponentFactoryDict",
    "ComponentOrPath",
    "ComponentOrReference",
    "ComponentSpec",
    "Coordinate",
    "Coordinates",
    "CrossSection",
    "CrossSectionFactory",
    "CrossSectionOrFactory",
    "CrossSectionSpec",
    "Float2",
    "Float3",
    "Floats",
    "Int2",
    "Int3",
    "Ints",
    "Label",
    "Layer",
    "LayerMap",
    "LayerLevel",
    "LayerSpec",
    "LayerSpecs",
    "LayerStack",
    "Layers",
    "MultiCrossSectionAngleSpec",
    "NameToFunctionDict",
    "Number",
    "Optional",
    "PathType",
    "PathTypes",
    "Route",
    "RouteFactory",
    "Routes",
    "Section",
    "Strs",
    "WidthTypes",
    "Union",
    "List",
    "Tuple",
    "Dict",
)


def write_schema(model: BaseModel = Netlist) -> None:
    from gdsfactory.config import PATH

    s = model.model_json_schema()
    d = OmegaConf.create(s)

    schema_path_json = PATH.schema_netlist
    schema_path_yaml = schema_path_json.with_suffix(".yaml")

    schema_path_yaml.write_text(OmegaConf.to_yaml(d))
    schema_path_json.write_text(json.dumps(OmegaConf.to_container(d)))


def _demo() -> None:
    write_schema()

    import jsonschema
    import yaml

    from gdsfactory.config import PATH

    schema_path_json = PATH.schema_netlist
    schema_dict = json.loads(schema_path_json.read_text())

    yaml_text = """

name: pads

instances:
    bl:
      component: pad
    tl:
      component: pad
    br:
      component: pad
    tr:
      component: pad

placements:
    tl:
        x: -200
        y: 500

    br:
        x: 400
        y: 400

    tr:
        x: 400
        y: 600


routes:
    electrical:
        settings:
            separation: 20
            width: 10
            path_length_match_loops: 2
            end_straight_length: 100
        links:
            tl,e3: tr,e1
            bl,e3: br,e1
    optical:
        settings:
            radius: 100
        links:
            bl,e4: br,e3
"""

    yaml_dict = yaml.safe_load(yaml_text)
    jsonschema.validate(yaml_dict, schema_dict)

    # from gdsfactory.components import factory
    # c = Netlist(factory=factory)
    # c.add_instance("mmi1", "mmi1x2", length=13.3)


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.mzi()
    s = Schematic()
    s.from_component(c)
    s.plot_netlist()
    # plt.show()
