"""Common data types.

In programming, a factory is a function that returns an Object.

Functions are easy to understand because they have clear inputs and outputs.
Most gdsfactory functions take some inputs and return a Component object.
Some of these inputs are other functions.

A Component is an object that has:

- name
- references to other components (x, y, rotation)
- polygons in different layers
- ports dictionary


A ComponentFactory is a function that returns a Component.


A Route is a dictionary with 3 keys:
- references: list of references (waveguides, bends and tapers)
- ports dictionary: dict(input=PortIn, output=PortOut)
- length: float (how long is this route)

A RouteFactory is a function that returns a Route.

"""

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union

from numpy import float64, int64, ndarray

from pp.component import Component, ComponentReference
from pp.port import Port

Layer = Tuple[int, int]
Layers = Iterable[Layer]
Route = Dict[str, Union[List[ComponentReference], Dict[str, Port], float]]
RouteFactory = Callable[..., Route]
ComponentFactory = Callable[..., Component]
PathType = Union[str, Path]

ComponentOrFactory = Union[ComponentFactory, Component]
ComponentOrPath = Union[PathType, Component]
ComponentOrReference = Union[Component, ComponentReference]
NameToFunctionDict = Dict[str, ComponentFactory]
Number = Union[float64, int64, float, int]
Coordinate = Union[Tuple[Number, Number], ndarray, List[Number]]
Coordinates = Union[
    List[Coordinate], ndarray, List[Number], Tuple[Number, ...], List[ndarray]
]
ComponentOrPath = Union[Component, PathType]


def get_name_to_function_dict(*functions) -> Dict[str, Callable]:
    """Returns a dict with function name as key and function as value."""
    return {func.__name__: func for func in functions}


__all__ = [
    "ComponentFactory",
    "ComponentOrFactory",
    "ComponentOrPath",
    "ComponentOrReference",
    "Coordinate",
    "Coordinates",
    "Layer",
    "Layers",
    "NameToFunctionDict",
    "Number",
    "PathType",
    "Route",
    "RouteFactory",
    "get_name_to_function_dict",
]
