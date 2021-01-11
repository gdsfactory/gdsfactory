"""Common data types."""

from typing import Callable, Dict, Tuple, Union

from phidl.device_layout import Device as Component

Layer = Tuple[int, int]
ComponentOrFunction = Union[Callable, Component]
NameToFunctionDict = Dict[str, Callable]
Real = Union[float, int]

__all__ = ["Layer", "ComponentOrFunction", "NameToFunctionDict", "Real"]
