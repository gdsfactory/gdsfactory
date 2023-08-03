from __future__ import annotations

import functools
import pathlib

import gdstk
import numpy as np

from gdsfactory.serialization import clean_value_json


def test_clean_value_json_bool() -> None:
    """Tests that boolean values are correctly serialized."""
    assert clean_value_json(True)
    assert not clean_value_json(False)


# Tests that nested dictionaries are correctly cleaned recursively.
def test_clean_value_json_recursive() -> None:
    nested_dict = {"a": {"b": {"c": 1}}}
    cleaned_dict = clean_value_json(nested_dict)
    assert cleaned_dict == {"a": {"b": {"c": 1}}}


# Tests that the function clean_value_json properly converts a numpy array to a list of lists of floats.
def test_clean_value_json_numpy_array() -> None:
    arr: np.ndarray = np.array([1.23456789, 2.34567891])
    expected: list[list[float]] = [[1.235, 2.346]]
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
    assert clean_value_json(partial_func) == expected, clean_value_json(partial_func)


# Tests that the clean_value_json function correctly serializes a gdstk.Polygon object by rounding its points to 3 decimal places.
def test_clean_value_json_gdstk_polygon() -> None:
    polygon: gdstk.Polygon = gdstk.Polygon([(0, 0), (1.23456789, 2.34567891)])
    expected: list[list[float]] = [[0, 0], [1.235, 2.346]]
    assert np.all(np.equal(clean_value_json(polygon), expected))


def test_clean_value_json():
    # Test boolean value
    assert clean_value_json(True) is True

    # Test integer value
    assert clean_value_json(10) == 10

    # Test float value
    assert clean_value_json(10.1) == round(10.1, 3)

    # Test numpy integer value
    assert clean_value_json(np.int64(10)) == 10

    # Test numpy float value
    assert clean_value_json(np.float64(10.1)) == round(10.1, 3)

    # Test numpy array value
    np_array = np.array([1, 2, 3])
    assert np.array_equal(clean_value_json(np_array), np_array.tolist())

    # Test callable function
    def test_func():
        pass

    assert clean_value_json(test_func) == {"function": "test_func"}

    # Test dictionary value
    test_dict = {"key": "value"}
    assert clean_value_json(test_dict) == test_dict

    # Test list value
    test_list = [1, 2, "3"]
    assert clean_value_json(test_list) == test_list

    # Test pathlib.Path value
    test_path = pathlib.Path("/tmp/test.txt")
    assert clean_value_json(test_path) == "test"

    # # Test unsupported type
    # class Unsupported: pass
    # unsupported = Unsupported()
    # with pytest.raises(TypeError):
    #     clean_value_json(unsupported)


if __name__ == "__main__":
    test_clean_value_json()
    # test_clean_value_json_callable()
    # test_clean_value_json_gdstk_polygon()
    # test_clean_value_json_numpy_array()
    # def func(a: int, b: int) -> int:
    #     return a + b

    # partial_func = functools.partial(func, b=2)
    # expected = {"function": "func", "settings": {"b": 2}}
    # assert clean_value_json(partial_func) == expected
