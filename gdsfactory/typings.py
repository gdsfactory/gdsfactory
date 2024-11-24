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
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Generator, Literal, ParamSpec, TypeAlias, TypeVar

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
from gdsfactory.cross_section import (  # type: ignore[attr-defined]
    CrossSection,
    Transition,
)
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


Anchor: TypeAlias = Literal[
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
Axis: TypeAlias = Literal["x", "y"]
NSEW: TypeAlias = Literal["N", "S", "E", "W"]
WidthTypes: TypeAlias = Literal["sine", "linear", "parabolic"]

Float2: TypeAlias = tuple[float, float]
Float3: TypeAlias = tuple[float, float, float]
Floats: TypeAlias = Sequence[float]
Strs: TypeAlias = Sequence[str]
Int2: TypeAlias = tuple[int, int]
Int3: TypeAlias = tuple[int, int, int]
Ints: TypeAlias = tuple[int, ...] | list[int]

BoundingBox: TypeAlias = tuple[float, float, float, float]
BoundingBoxes: TypeAlias = Sequence[BoundingBox]
Size: TypeAlias = tuple[float, float]
Spacing: TypeAlias = tuple[float, float]
Radius: TypeAlias = float

Delta: TypeAlias = float
AngleInDegrees: TypeAlias = float

Layer: TypeAlias = tuple[int, int]
Layers: TypeAlias = Sequence[Layer]
LayerSpec: TypeAlias = LayerEnum | str | tuple[int, int]
LayerSpecs: TypeAlias = Sequence[LayerSpec]

AnyComponent: TypeAlias = Component | ComponentAllAngle
AnyComponentT = TypeVar("AnyComponentT", bound=AnyComponent)
AnyComponentFactory: TypeAlias = Callable[..., AnyComponent]
AnyComponentPostProcess: TypeAlias = Callable[[AnyComponent], None]

ComponentParams = ParamSpec("ComponentParams")
ComponentFactory: TypeAlias = Callable[..., Component]
ComponentAllAngleFactory: TypeAlias = Callable[..., ComponentAllAngle]
ComponentBaseFactory: TypeAlias = Callable[..., ComponentBase]
ComponentFactoryDict: TypeAlias = dict[str, ComponentFactory]

PathType: TypeAlias = str | pathlib.Path
PathTypes: TypeAlias = Sequence[PathType]
Metadata: TypeAlias = dict[str, int | float | str]
PostProcess: TypeAlias = Callable[[Component], None]
PostProcesses: TypeAlias = Sequence[PostProcess]
MaterialSpec: TypeAlias = str | float | tuple[float, float] | Callable[..., Any]

Instance: TypeAlias = ComponentReference
ComponentOrPath: TypeAlias = PathType | Component
ComponentOrReference: TypeAlias = Component | ComponentReference
NameToFunctionDict: TypeAlias = dict[str, ComponentFactory]
Number: TypeAlias = float | int
Coordinate: TypeAlias = tuple[float, float]
Coordinates: TypeAlias = Sequence[Coordinate]
CrossSectionFactory: TypeAlias = Callable[..., CrossSection]
CrossSectionOrFactory: TypeAlias = CrossSection | Callable[..., CrossSection]

WidthFunction: TypeAlias = Callable[..., npt.NDArray[np.float64]]
OffsetFunction: TypeAlias = Callable[..., npt.NDArray[np.float64]]

Port: TypeAlias = kf.Port
PortFactory: TypeAlias = Callable[..., Port]
PortsFactory: TypeAlias = Callable[..., Sequence[Port]]
PortSymmetries: TypeAlias = dict[str, Sequence[str]]
PortsDict: TypeAlias = dict[str, Port]
Ports: TypeAlias = kf.Ports | Sequence[Port] | Iterable[Port]
SelectPorts: TypeAlias = Callable[..., Sequence[Port]]

PortType: TypeAlias = str
PortName: TypeAlias = str

PortTypes: TypeAlias = Sequence[PortType]
PortNames: TypeAlias = Sequence[PortName]

Sparameters: TypeAlias = dict[str, npt.NDArray[np.float64]]

ComponentSpec: TypeAlias = str | ComponentFactory | dict[str, Any] | kf.KCell
ComponentSpecOrComponent: TypeAlias = (
    str | ComponentFactory | dict[str, Any] | kf.KCell | Component
)

ComponentSpecs: TypeAlias = Sequence[ComponentSpec]
ComponentSpecsOrComponents: TypeAlias = Sequence[ComponentSpecOrComponent]
ComponentFactories: TypeAlias = Sequence[ComponentFactory]

ComponentSpecOrList: TypeAlias = ComponentSpec | list[ComponentSpec]
CellSpec: TypeAlias = (
    str | ComponentFactory | dict[str, Any]  # PCell function, function name or dict
)

ComponentSpecDict: TypeAlias = dict[str, ComponentSpec]
CrossSectionSpec: TypeAlias = CrossSectionFactory | CrossSection | dict[str, Any] | str
CrossSectionSpecs: TypeAlias = tuple[CrossSectionSpec, ...]

MultiCrossSectionAngleSpec: TypeAlias = list[tuple[CrossSectionSpec, tuple[int, ...]]]


ConductorConductorName: TypeAlias = tuple[str, str]
ConductorViaConductorName: TypeAlias = tuple[str, str, str] | tuple[str, str]
ConnectivitySpec: TypeAlias = ConductorConductorName | ConductorViaConductorName

Route: TypeAlias = (
    kf.routing.generic.ManhattanRoute | kf.routing.aa.optical.OpticalAllAngleRoute
)
RoutingStrategy: TypeAlias = Callable[..., Sequence[Route]]
RoutingStrategies: TypeAlias = dict[str, RoutingStrategy]


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
    "AnyComponent",
    "AnyComponentFactory",
    "AnyComponentT",
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
    "Port",
    "PortName",
    "PortNames",
    "PortType",
    "PortTypes",
    "Ports",
    "PostProcesses",
    "Radius",
    "RoutingStrategies",
    "SelectPorts",
    "Size",
    "Spacing",
    "Strs",
    "Transition",
    "WidthTypes",
)
