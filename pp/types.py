"""Common data types."""

from typing import Callable, Dict, Tuple, Union

from pp.component import Component

Layer = Tuple[int, int]
ComponentOrFunction = Union[Callable, Component]
NameToFunctionDict = Dict[str, Callable]
Number = Union[float, int]

__all__ = ["Layer", "ComponentOrFunction", "NameToFunctionDict", "Number"]
