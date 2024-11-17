"""In programming, a factory is a function that returns an object.

Functions are easy to understand because they have clear inputs and outputs.
Most gdsfactory functions take some inputs and return a Component object.
Some of these inputs parameters are also functions.

- Component: Object with.
    - name.
    - references: to other components (x, y, rotation).
    - polygons in different layers.
    - ports dict.


Specs:

- ComponentSpec: Component, function, string or dict
    (component=mzi, settings=dict(delta_length=20)).
- LayerSpec: (3, 0), 3 (assumes 0 as datatype) or string.

"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable, Sequence
from typing import (
    Any,
    Generator,
    Literal,
    ParamSpec,
    TypeAlias,
)

import kfactory as kf
import numpy as np
import numpy.typing as npt
from kfactory.kcell import LayerEnum

from gdsfactory.component import (
    Component,
    ComponentAllAngle,
    ComponentBase,
    ComponentReference,
)
from gdsfactory.cross_section import CrossSection, Section, Transition, WidthTypes
from gdsfactory.port import Port
from gdsfactory.technology import LayerLevel, LayerMap, LayerStack, LayerViews

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
    dx: Delta | None = None
    dy: Delta | None = None


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


Float2: TypeAlias = tuple[float, float]
Float3: TypeAlias = tuple[float, float, float]
Floats: TypeAlias = tuple[float, ...] | list[float]
Strs: TypeAlias = tuple[str, ...] | list[str]
Int2: TypeAlias = tuple[int, int]
Int3: TypeAlias = tuple[int, int, int]
Ints: TypeAlias = tuple[int, ...] | list[int]

Size: TypeAlias = tuple[float, float]
Position: TypeAlias = tuple[float, float]
Spacing: TypeAlias = tuple[float, float]
Radius: TypeAlias = float

Delta: TypeAlias = float
AngleInDegrees: TypeAlias = float

Layer: TypeAlias = tuple[int, int]
Layers: TypeAlias = tuple[Layer, ...] | list[Layer]
LayerSpec: TypeAlias = LayerEnum | str | tuple[int, int]
LayerSpecs: TypeAlias = list[LayerSpec] | tuple[LayerSpec, ...]

ComponentParams = ParamSpec("ComponentParams")
ComponentFactory: TypeAlias = Callable[..., Component]
ComponentFactoryDict: TypeAlias = dict[str, ComponentFactory]
PathType: TypeAlias = str | pathlib.Path
PathTypes: TypeAlias = tuple[PathType, ...]
Metadata: TypeAlias = dict[str, int | float | str]
PostProcess: TypeAlias = tuple[Callable[[Component], None], ...]

MaterialSpec: TypeAlias = str | float | tuple[float, float] | Callable[..., Any]

Instance = ComponentReference
ComponentOrPath: TypeAlias = PathType | Component
ComponentOrReference: TypeAlias = Component | ComponentReference
NameToFunctionDict: TypeAlias = dict[str, ComponentFactory]
Number: TypeAlias = float | int
Coordinate: TypeAlias = tuple[float, float]
Coordinates: TypeAlias = tuple[Coordinate, ...] | list[Coordinate]
CrossSectionFactory: TypeAlias = Callable[..., CrossSection]
TransitionFactory: TypeAlias = Callable[..., Transition]
CrossSectionOrFactory: TypeAlias = CrossSection | Callable[..., CrossSection]
PortSymmetries: TypeAlias = dict[str, list[str]]
PortsDict: TypeAlias = dict[str, Port]
PortsList: TypeAlias = list[Port]
Ports = kf.Ports
PortsOrList: TypeAlias = Ports | PortsList

Sparameters: TypeAlias = dict[str, npt.NDArray[np.float64]]

ComponentSpec: TypeAlias = (
    str | ComponentFactory | dict[str, Any] | kf.KCell
)  # PCell function, function name, dict or Component
ComponentSpecOrComponent: TypeAlias = (
    str | ComponentFactory | dict[str, Any] | kf.KCell | Component
)  # PCell function, function name, dict or Component

ComponentSpecs: TypeAlias = tuple[ComponentSpec, ...]
ComponentSpecsOrComponents: TypeAlias = Sequence[ComponentSpecOrComponent]
ComponentFactories: TypeAlias = tuple[ComponentFactory, ...]

ComponentSpecOrList: TypeAlias = ComponentSpec | list[ComponentSpec]
CellSpec: TypeAlias = (
    str | ComponentFactory | dict[str, Any]
)  # PCell function, function name or dict

ComponentSpecDict: TypeAlias = dict[str, ComponentSpec]
CrossSectionSpec: TypeAlias = (
    CrossSectionFactory | CrossSection | dict[str, Any] | str | Transition
)
CrossSectionSpecs: TypeAlias = tuple[CrossSectionSpec, ...]

MultiCrossSectionAngleSpec: TypeAlias = list[tuple[CrossSectionSpec, tuple[int, ...]]]


ConductorConductorName: TypeAlias = tuple[str, str]
ConductorViaConductorName: TypeAlias = tuple[str, str, str] | tuple[str, str]
ConnectivitySpec: TypeAlias = ConductorConductorName | ConductorViaConductorName

_RoutingStrategy: TypeAlias = Callable[
    ...,
    list[kf.routing.generic.ManhattanRoute]
    | list[kf.routing.aa.optical.OpticalAllAngleRoute],
]
RoutingStrategies: TypeAlias = dict[str, _RoutingStrategy]


class TypedArray(np.ndarray[Any, np.dtype[Any]]):
    """based on https://github.com/samuelcolvin/pydantic/issues/380."""

    @classmethod
    def __get_validators__(
        cls,
    ) -> Generator[Callable[[Any, Any], npt.NDArray[np.float64]], Any, None]:
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val: Any, _info: Any) -> npt.NDArray[np.float64]:
        return np.array(val, dtype=cls.inner_type)  # type: ignore


class ArrayMeta(type):
    def __getitem__(cls, t: np.dtype[Any]) -> type[npt.NDArray[Any]]:
        return type("Array", (TypedArray,), {"inner_type": t})


class Array(np.ndarray[Any, np.dtype[Any]], metaclass=ArrayMeta):
    pass


__all__ = (
    "AngleInDegrees",
    "Any",
    "Component",
    "ComponentAllAngle",
    "ComponentBase",
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
    "Delta",
    "Float2",
    "Float3",
    "Floats",
    "Instance",
    "Int2",
    "Int3",
    "Ints",
    "Layer",
    "LayerLevel",
    "LayerMap",
    "LayerSpec",
    "LayerSpecs",
    "LayerStack",
    "LayerViews",
    "Layers",
    "MultiCrossSectionAngleSpec",
    "NameToFunctionDict",
    "Number",
    "PathType",
    "PathTypes",
    "Ports",
    "PortsList",
    "PortsOrList",
    "Position",
    "Radius",
    "RoutingStrategies",
    "Section",
    "Size",
    "Spacing",
    "Strs",
    "WidthTypes",
)
