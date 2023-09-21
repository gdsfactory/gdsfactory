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
from pydantic import BaseModel

from gdsfactory.component import Component, ComponentReference
from gdsfactory.component_layout import Label
from gdsfactory.cross_section import CrossSection, Section, Transition, WidthTypes
from gdsfactory.port import Port
from gdsfactory.technology import LayerLevel, LayerStack

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

LayerSpec = (
    Layer | int | str
)  # tuple of integers (layer, datatype), a integer (layer, 0) or a string (layer_name)

LayerSpecs = list[LayerSpec] | tuple[LayerSpec, ...] | None
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

Sparameters = dict[str, np.ndarray]

ComponentSpec = (
    str | ComponentFactory | Component | dict[str, Any]
)  # PCell function, function name, dict or Component

ComponentSpecOrList = ComponentSpec | list[ComponentSpec]
CellSpec = (
    str | ComponentFactory | dict[str, Any]
)  # PCell function, function name or dict

ComponentSpecDict = dict[str, ComponentSpec]
CrossSectionSpec = (
    str
    | CrossSectionFactory
    | CrossSection
    | Transition
    | TransitionFactory
    | dict[str, Any]
)  # cross_section function, function name or dict
CrossSectionSpecs = tuple[CrossSectionSpec, ...]

MultiCrossSectionAngleSpec = list[tuple[CrossSectionSpec, tuple[int, ...]]]

LabelListFactory = Callable[..., list[Label]]


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


class ComponentModel(BaseModel):
    component: str | dict[str, Any]
    settings: dict[str, Any] | None

    model_config = {"extra": "forbid"}


class PlacementModel(BaseModel):
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

    model_config = {"extra": "forbid"}


class RouteModel(BaseModel):
    links: dict[str, str]
    settings: dict[str, Any] | None = None
    routing_strategy: str | None = None

    model_config = {"extra": "forbid"}


class NetlistModel(BaseModel):
    """Netlist defined component.

    Parameters:
        instances: dict of instances (name, settings, component).
        placements: dict of placements.
        connections: dict of connections.
        routes: dict of routes.
        name: component name.
        info: information (polarization, wavelength ...).
        settings: input variables.
        ports: exposed component ports.

    """

    instances: dict[str, ComponentModel] | None = None
    placements: dict[str, PlacementModel] | None = None
    connections: dict[str, str] | None = None
    routes: dict[str, RouteModel] | None = None
    name: str | None = None
    info: dict[str, Any] | None = None
    settings: dict[str, Any] | None = None
    ports: dict[str, str] | None = None

    model_config = {"extra": "forbid"}


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


def write_schema(model: BaseModel = NetlistModel) -> None:
    from gdsfactory.config import PATH

    s = model.schema_json()
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

name: mzi

pdk: ubcpdk

settings:
   dy: -90

info:
    polarization: te
    wavelength: 1.55
    description: mzi for ubcpdk

instances:
    yr:
      component: y_splitter
    yl:
      component: y_splitter

placements:
    yr:
        rotation: 180
        x: 100
        y: 0

routes:
    route_top:
        links:
            yl,opt2: yr,opt3
        settings:
            cross_section: strip
    route_bot:
        links:
            yl,opt3: yr,opt2
        routing_strategy: get_bundle_from_steps
        settings:
          steps: [dx: 30, dy: '${settings.dy}', dx: 20]
          cross_section: strip

ports:
    o1: yl,opt1
    o2: yr,opt1
"""

    yaml_dict = yaml.safe_load(yaml_text)
    jsonschema.validate(yaml_dict, schema_dict)

    # from gdsfactory.components import factory
    # c = NetlistModel(factory=factory)
    # c.add_instance("mmi1", "mmi1x2", length=13.3)


if __name__ == "__main__":
    s = Step()
