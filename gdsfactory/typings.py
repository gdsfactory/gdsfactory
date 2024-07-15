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
from collections.abc import Callable, Iterable
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    ParamSpec,
    Tuple,
    TypeAlias,
    Union,
)

import kfactory as kf
import numpy as np
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
    dx: float | None = None
    dy: float | None = None


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

Layer = LayerEnum
Layers = tuple[Layer, ...]
LayerSpec = LayerEnum | str | tuple[int, int]

LayerSpecs = list[LayerSpec] | tuple[LayerSpec, ...]

ComponentParams = ParamSpec("ComponentParams")
ComponentFactory = Callable[..., Component]
ComponentFactoryDict = dict[str, ComponentFactory]
PathType = str | pathlib.Path
PathTypes = tuple[PathType, ...]
Metadata = dict[str, int | float | str]
PostProcess = tuple[Callable[[Component], None], ...]


MaterialSpec = str | float | tuple[float, float] | Callable

ComponentOrPath = PathType | Component
ComponentOrReference = Component | ComponentReference
NameToFunctionDict = dict[str, ComponentFactory]
Number = float | int
Coordinate = tuple[float, float]
Coordinates = tuple[Coordinate, ...] | list[Coordinate]
CrossSectionFactory = Callable[..., CrossSection]
TransitionFactory = Callable[..., Transition]
CrossSectionOrFactory = CrossSection | Callable[..., CrossSection]
PortSymmetries = dict[str, list[str]]
PortsDict = dict[str, Port]
PortsList = dict[str, Port]

Sparameters = dict[str, np.ndarray]

ComponentSpec = (
    str | ComponentFactory | dict[str, Any] | kf.KCell
)  # PCell function, function name, dict or Component

ComponentSpecs = tuple[ComponentSpec, ...]
ComponentFactories = tuple[ComponentFactory, ...]

ComponentSpecOrList = ComponentSpec | list[ComponentSpec]
CellSpec = (
    str | ComponentFactory | dict[str, Any]
)  # PCell function, function name or dict

ComponentSpecDict = dict[str, ComponentSpec]
CrossSectionSpec: TypeAlias = (
    CrossSectionFactory | CrossSection | dict[str, Any] | str | Transition
)
CrossSectionSpecs = tuple[CrossSectionSpec, ...]

MultiCrossSectionAngleSpec = list[tuple[CrossSectionSpec, tuple[int, ...]]]


ConductorConductorName = tuple[str, str]
ConductorViaConductorName = tuple[str, str, str] | tuple[str, str]
ConnectivitySpec = ConductorConductorName | ConductorViaConductorName


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
    "Float2",
    "Float3",
    "Floats",
    "Int2",
    "Int3",
    "Ints",
    "Layer",
    "LayerMap",
    "LayerLevel",
    "LayerSpec",
    "LayerSpecs",
    "LayerStack",
    "LayerViews",
    "Layers",
    "MultiCrossSectionAngleSpec",
    "NameToFunctionDict",
    "Number",
    "Optional",
    "PathType",
    "PathTypes",
    "Section",
    "Strs",
    "WidthTypes",
    "Union",
    "List",
    "Tuple",
    "Dict",
    "Iterable",
)
