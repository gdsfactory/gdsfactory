"""Common data types."""

from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

from pp.component import Component, ComponentReference
from pp.port import Port

Layer = Tuple[int, int]
ComponentOrFunction = Union[Callable, Component]
ComponentOrPath = Union[Path, Component]
NameToFunctionDict = Dict[str, Callable]
Number = Union[float, int]
Route = Dict[str, Union[List[ComponentReference], Dict[str, Port], float]]


def get_name_to_function_dict(*functions) -> Dict[str, Callable]:
    """Returns a dict with function name as key and function as value"""
    return {func.__name__: func for func in functions}


__all__ = [
    "Layer",
    "ComponentOrFunction",
    "NameToFunctionDict",
    "Number",
    "get_name_to_function_dict",
]
