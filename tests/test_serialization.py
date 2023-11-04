from __future__ import annotations

import functools
import pathlib

import gdstk
import numpy as np

import gdsfactory as gf
from gdsfactory.cross_section import strip
from gdsfactory.serialization import clean_value_json


@gf.cell
def demo_cross_section_setting(cross_section=strip) -> gf.Component:
    return gf.components.straight(cross_section=cross_section)


@gf.cell
def demo_dict_keys(port_names: tuple[str, ...]):
    return gf.Component()


def test_settings(data_regression, check: bool = True) -> None:
    """Avoid regressions when exporting settings."""
    component = demo_cross_section_setting()
    settings = component.to_dict()
    if data_regression:
        data_regression.check(settings)


@gf.cell
def wrap_polygon(polygon) -> gf.Component:
    return gf.Component()


@gf.cell
def wrap_polygons(polygons) -> gf.Component:
    return gf.Component()


def test_serialize_polygons() -> None:
    wrap_polygon(gdstk.rectangle((0, 0), (1, 1)))  # FAILS

    s = gf.components.straight()
    wrap_polygons(s.get_polygons(as_array=False))  # FAILS
    wrap_polygons(s.get_polygons(by_spec=False, as_array=True))  # WORKS
    wrap_polygons(s.get_polygons(by_spec=True, as_array=True))  # WORKS

    s = gf.components.ring_double_heater()
    wrap_polygons(s.get_polygons(by_spec=False, as_array=True))  # FAILS
    wrap_polygons(s.get_polygons(by_spec=(1, 0), as_array=True))  # FAILS
    wrap_polygons(s.get_polygons(by_spec=True, as_array=True))  # FAILS


def test_serialize_dict_keys():
    c1 = gf.c.straight()
    demo_dict_keys(port_names=c1.ports.keys())


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

    # Test complex value
    assert clean_value_json(1.0 + 1j) == {"real": round(1.0, 3), "imag": round(1.0, 3)}

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
