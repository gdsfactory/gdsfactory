import pathlib

import gdsfactory as gf
from gdsfactory.export.to_stl import to_stl


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
    component = gf.c.pad()
    filepath = "test.stl"
    exclude_layers = [(49, 0)]
    to_stl(component, filepath, exclude_layers=exclude_layers)
    filepath = "test_49_0.stl"
    assert not pathlib.Path(filepath).exists()


if __name__ == "__main__":
    test_export_filepath()
    # test_export_empty_component()
    # component = Component()
    # filepath = 'test.stl'
    # to_stl(component, filepath)
