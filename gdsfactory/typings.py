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
from collections.abc import Callable, Generator, Sequence
from functools import partial
from typing import Any, Literal, ParamSpec, Protocol, TypeAlias, TypeVar

import kfactory as kf
import klayout.db as kdb
import numpy as np
import numpy.typing as npt
from kfactory.layer import LayerEnum

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
Position: TypeAlias = tuple[float, float]
Radius: TypeAlias = float

Delta: TypeAlias = float
AngleInDegrees: TypeAlias = float

Layer: TypeAlias = tuple[int, int]
Layers: TypeAlias = Sequence[Layer]
LayerSpec: TypeAlias = Layer | str | int | LayerEnum
LayerSpecs: TypeAlias = Sequence[LayerSpec]

Number: TypeAlias = float | int
Coordinate: TypeAlias = tuple[float, float]
Coordinates: TypeAlias = Sequence[Coordinate]
WayPoints: TypeAlias = Sequence[Coordinate | kdb.DPoint]

MaterialSpec: TypeAlias = (
    str | float | tuple[float, float] | Callable[..., Any] | npt.NDArray[np.float64]
)

WidthFunction: TypeAlias = Callable[..., npt.NDArray[np.floating[Any]]]
OffsetFunction: TypeAlias = Callable[[float], float]

PathType: TypeAlias = str | pathlib.Path
PathTypes: TypeAlias = Sequence[PathType]
Metadata: TypeAlias = dict[str, int | float | str]

Port: TypeAlias = kf.DPort
TPort = TypeVar("TPort", bound=Port)
IOPorts: TypeAlias = tuple[str, str]
PortFactory: TypeAlias = Callable[..., Port]
PortsFactory: TypeAlias = Callable[..., Sequence[Port]]
PortSymmetries: TypeAlias = dict[str, Sequence[str]]
PortDict: TypeAlias = dict[str, Port]
Ports: TypeAlias = kf.DPorts | Sequence[Port] | kf.DInstancePorts
SelectPorts: TypeAlias = Callable[..., Sequence[Port]]

PortType: TypeAlias = str
PortName: TypeAlias = str

PortTypes: TypeAlias = Sequence[PortType]
PortNames: TypeAlias = Sequence[PortName]

PortsDict: TypeAlias = dict[str, list[Port]]
PortsDictGeneric: TypeAlias = dict[str, list[TPort]]

ConductorConductorName: TypeAlias = tuple[str, str]
ConductorViaConductorName: TypeAlias = tuple[str, str, str] | ConductorConductorName
ConnectivitySpec: TypeAlias = ConductorConductorName | ConductorViaConductorName

Sparameters: TypeAlias = dict[str, npt.NDArray[np.float64]]

Route: TypeAlias = (
    kf.routing.generic.ManhattanRoute | kf.routing.aa.optical.OpticalAllAngleRoute
)
RoutingStrategy: TypeAlias = Callable[..., Sequence[Route]]
RoutingStrategies: TypeAlias = dict[str, RoutingStrategy]


from gdsfactory.cross_section import CrossSectionFactory, CrossSectionSpec  # noqa: E402

MultiCrossSectionAngleSpec: TypeAlias = Sequence[
    tuple[CrossSectionSpec, tuple[int, ...]]
]

from gdsfactory import component  # noqa: E402

AnyComponent: TypeAlias = component.Component | component.ComponentAllAngle
AnyComponentT = TypeVar("AnyComponentT", bound=AnyComponent)
AnyComponentFactory: TypeAlias = Callable[..., AnyComponent]
AnyComponentPostProcess: TypeAlias = (
    Callable[[component.Component], None]
    | Callable[[component.ComponentAllAngle], None]
)

ComponentParams = ParamSpec("ComponentParams")
ComponentFactory: TypeAlias = Callable[..., component.Component]
ComponentAllAngleFactory: TypeAlias = Callable[..., component.ComponentAllAngle]
ComponentBaseFactory: TypeAlias = Callable[..., component.ComponentBase]
ComponentFactoryDict: TypeAlias = dict[str, ComponentFactory]
ComponentFactories: TypeAlias = Sequence[ComponentFactory]

ComponentSpec: TypeAlias = str | ComponentFactory | dict[str, Any] | kf.DKCell
ComponentSpecOrComponent: TypeAlias = ComponentSpec | component.Component
ComponentSpecs: TypeAlias = Sequence[ComponentSpec]
ComponentSpecsOrComponents: TypeAlias = Sequence[ComponentSpecOrComponent]


class _PostProcess(Protocol):
    def __call__(self, component: component.Component, **kwargs: Any) -> Any: ...


PostProcess: TypeAlias = (
    _PostProcess | Callable[[component.Component], None] | partial[component.Component]
)
PostProcesses: TypeAlias = Sequence[PostProcess]

Instance: TypeAlias = component.ComponentReference
ComponentOrPath: TypeAlias = PathType | component.Component
ComponentOrReference: TypeAlias = component.Component | component.ComponentReference
NameToFunctionDict: TypeAlias = dict[str, ComponentFactory]


ComponentSpecOrList: TypeAlias = ComponentSpec | list[ComponentSpec]
CellSpec: TypeAlias = (
    str | ComponentFactory | dict[str, Any]  # PCell function, function name or dict
)
ComponentSpecDict: TypeAlias = dict[str, ComponentSpec]

LayerTransitions: TypeAlias = dict[LayerSpec | tuple[Layer, Layer], ComponentSpec]


class TypedArray(np.ndarray[Any, np.dtype[Any]]):
    """based on https://github.com/samuelcolvin/pydantic/issues/380."""

    @classmethod
    def __get_validators__(
        cls,
    ) -> Generator[Callable[[Any, Any], npt.NDArray[np.float64]], Any, None]:
        yield cls.validate_type  # pragma: no cover

    @classmethod
    def validate_type(cls, val: Any, _info: Any) -> npt.NDArray[np.float64]:
        return np.array(val, dtype=cls.inner_type)  # type: ignore[attr-defined]


class ArrayMeta(type):
    def __getitem__(cls, t: np.dtype[Any]) -> type[npt.NDArray[Any]]:
        return type("Array", (TypedArray,), {"inner_type": t})  # pragma: no cover


class Array(np.ndarray[Any, np.dtype[Any]], metaclass=ArrayMeta): ...


__all__ = (
    "AngleInDegrees",
    "ComponentSpec",
    "Coordinate",
    "Coordinates",
    "CrossSectionFactory",
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
    "LayerSpec",
    "LayerSpecs",
    "LayerTransitions",
    "Layers",
    "Number",
    "PathType",
    "PathTypes",
    "Port",
    "PortDict",
    "PortName",
    "PortNames",
    "PortType",
    "PortTypes",
    "Ports",
    "PortsDict",
    "PortsDictGeneric",
    "Position",
    "PostProcesses",
    "Radius",
    "RoutingStrategies",
    "SelectPorts",
    "Size",
    "Spacing",
    "Strs",
    "TPort",
    "WayPoints",
    "WidthTypes",
)
