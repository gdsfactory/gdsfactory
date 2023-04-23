from __future__ import annotations
from typing import List
import functools
import pathlib

import gdstk
import numpy as np

from gdsfactory.serialization import clean_value_json


def test_clean_value_json_bool():
    """Tests that boolean values are correctly serialized."""
    assert clean_value_json(True)
    assert not clean_value_json(False)


# Tests that nested dictionaries are correctly cleaned recursively.
def test_clean_value_json_recursive():
    nested_dict = {"a": {"b": {"c": 1}}}
    cleaned_dict = clean_value_json(nested_dict)
    assert cleaned_dict == {"a": {"b": {"c": 1}}}


# Tests that the function clean_value_json properly converts a numpy array to a list of lists of floats.
def test_clean_value_json_numpy_array() -> None:
    arr: np.ndarray = np.array([1.23456789, 2.34567891])
    expected: List[List[float]] = [[1.23456789, 2.34567891]]
    assert np.all(np.equal(clean_value_json(arr), expected))


# Tests that the function clean_value_json correctly extracts the stem of a pathlib.Path object.
def test_clean_value_json_pathlib_path() -> None:
    path: pathlib.Path = pathlib.Path("/path/to/file.txt")
    assert clean_value_json(path) == "file"


# Tests the functionality of the clean_value_json() function when given a callable object.
def test_clean_value_json_callable() -> None:
    def func(a: int, b: int) -> int:
        return a + b

    partial_func = functools.partial(func, b=2)
    expected = {"function": "func", "settings": {"b": 2}}
    assert clean_value_json(partial_func) == expected


# Tests that the clean_value_json function correctly serializes a gdstk.Polygon object by rounding its points to 3 decimal places.
def test_clean_value_json_gdstk_polygon() -> None:
    polygon: gdstk.Polygon = gdstk.Polygon([(0, 0), (1.23456789, 2.34567891)])
    expected: List[List[float]] = [[0, 0], [1.235, 2.346]]
    assert np.all(np.equal(clean_value_json(polygon), expected))


if __name__ == "__main__":
    # test_clean_value_json_gdstk_polygon()
    # test_clean_value_json_numpy_array()
    def func(a: int, b: int) -> int:
        return a + b

    partial_func = functools.partial(func, b=2)
    expected = {"function": "func", "settings": {"b": 2}}
    assert clean_value_json(partial_func) == expected
