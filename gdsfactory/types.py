"""In programming, a factory is a function that returns an Object.

Functions are easy to understand because they have clear inputs and outputs.
Most gdsfactory functions take some inputs and return a Component object.
Some of these inputs parameters are also functions.

- Component: Object with.
    - name
    - references to other components (x, y, rotation)
    - polygons in different layers
    - ports dictionary
- ComponentFactory: function that returns a Component.
- Route: dataclass with 3 attributes.
    - references: list of references (straights, bends and tapers)
    - ports: dict(input=PortIn, output=PortOut)
    - length: float (how long is this route)
- RouteFactory: function that returns a Route.

"""
import json
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from omegaconf import OmegaConf
from phidl.device_layout import Label as LabelPhidl
from phidl.device_layout import Path
from pydantic import BaseModel
from typing_extensions import Literal

from gdsfactory.component import Component, ComponentReference
from gdsfactory.cross_section import CrossSection
from gdsfactory.port import Port

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


class Label(LabelPhidl):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """check with pydantic Label valid type"""
        assert isinstance(v, LabelPhidl), f"TypeError, Got {type(v)}, expecting Label"
        return v


class Route(BaseModel):
    references: List[ComponentReference]
    labels: Optional[List[Label]] = None
    ports: Tuple[Port, Port]
    length: float


class Routes(BaseModel):
    references: List[ComponentReference]
    lengths: List[float]
    ports: Optional[List[Port]] = None
    bend_radius: Optional[List[float]] = None


class ComponentModel(BaseModel):
    component: str
    settings: Optional[Dict[str, Any]]


class PlacementModel(BaseModel):
    x: Union[str, float] = 0
    y: Union[str, float] = 0
    dx: float = 0
    dy: float = 0
    port: Optional[Union[str, Anchor]] = None
    rotation: int = 0
    mirror: bool = False


class RouteModel(BaseModel):
    links: Dict[str, str]
    settings: Optional[Dict[str, Any]] = None
    routing_strategy: Optional[str] = None


class CircuitModel(BaseModel):
    instances: Dict[str, ComponentModel]
    name: Optional[str] = None
    placements: Optional[Dict[str, PlacementModel]] = None
    connections: Optional[List[Dict[str, str]]] = None
    routes: Optional[Dict[str, RouteModel]] = None
    info: Optional[Dict[str, Any]] = None


Float2 = Tuple[float, float]
Float3 = Tuple[float, float, float]
Floats = Tuple[float, ...]
Strs = Tuple[str, ...]
Int2 = Tuple[int, int]
Int3 = Tuple[int, int, int]
Ints = Tuple[int, ...]

Layer = Tuple[int, int]
Layers = Tuple[Layer, ...]
RouteFactory = Callable[..., Route]
ComponentFactory = Callable[..., Component]
ComponentFactoryDict = Dict[str, ComponentFactory]
PathFactory = Callable[..., Path]
PathType = Union[str, pathlib.Path]
PathTypes = Tuple[PathType, ...]

ComponentOrFactory = Union[ComponentFactory, Component]
ComponentOrFactoryOrList = Union[ComponentOrFactory, List[ComponentOrFactory]]
ComponentOrPath = Union[PathType, Component]
ComponentOrReference = Union[Component, ComponentReference]
NameToFunctionDict = Dict[str, ComponentFactory]
Number = Union[float, int]
Coordinate = Tuple[float, float]
Coordinates = Tuple[Coordinate, ...]
ComponentOrPath = Union[Component, PathType]
CrossSectionFactory = Callable[..., CrossSection]
CrossSectionOrFactory = Union[CrossSection, Callable[..., CrossSection]]


__all__ = (
    "ComponentFactory",
    "ComponentFactoryDict",
    "ComponentOrFactory",
    "ComponentOrPath",
    "ComponentOrReference",
    "Coordinate",
    "Coordinates",
    "CrossSectionFactory",
    "CrossSectionOrFactory",
    "Float2",
    "Float3",
    "Floats",
    "Int2",
    "Int3",
    "Ints",
    "Layer",
    "Layers",
    "NameToFunctionDict",
    "Number",
    "PathType",
    "PathTypes",
    "Route",
    "RouteFactory",
    "Routes",
    "Strs",
)


def write_schema(model: BaseModel = CircuitModel):
    s = model.schema_json()
    d = OmegaConf.create(s)

    f1 = pathlib.Path(__file__).parent / "schema.yaml"
    f1.write_text(OmegaConf.to_yaml(d))

    f2 = pathlib.Path(__file__).parent / "schema.json"
    f2.write_text(json.dumps(OmegaConf.to_container(d)))


if __name__ == "__main__":
    write_schema()
