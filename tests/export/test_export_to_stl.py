import pathlib

import pytest

import gdsfactory as gf
from gdsfactory.export.to_stl import to_stl
from gdsfactory.generic_tech.layer_map import LAYER


# Tests that a Component is exported into STL with a specified filepath prefix.
def test_export_filepath() -> None:
    component = gf.c.pad()
    filepath = "test.stl"
    to_stl(component, filepath)
    filepath = "test_49_0.stl"
    assert pathlib.Path(filepath).exists()
    pathlib.Path(filepath).unlink()


# Tests that a Component is exported into STL with a specified layer stack.
def test_export_scale() -> None:
    component = gf.c.pad()
    filepath = "test.stl"
    to_stl(component, filepath, scale=2)
    filepath = "test_49_0.stl"
    assert pathlib.Path(filepath).exists()
    pathlib.Path(filepath).unlink()


# Tests that a Component is exported into STL with a specified excluded layer.
def test_export_exclude_layers() -> None:
    component = gf.c.pad(layer=LAYER.M3)
    filepath = "test.stl"
    exclude_layers = [LAYER.M3]
    with pytest.raises(ValueError):
        to_stl(component, filepath, exclude_layers=exclude_layers)
        filepath = "test_49_0.stl"
        assert not pathlib.Path(filepath).exists()
