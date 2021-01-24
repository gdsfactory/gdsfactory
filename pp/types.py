"""Common data types."""

from typing import Callable, Dict, Tuple, Union

from pp.component import Component

Layer = Tuple[int, int]
ComponentOrFunction = Union[Callable, Component]
NameToFunctionDict = Dict[str, Callable]
Number = Union[float, int]


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
