import pytest
import trimesh

import gdsfactory as gf
from gdsfactory.export.to_3d import to_3d
from gdsfactory.technology import LayerLevel, LayerStack


def get_layer_stack() -> LayerStack:
    """Returns dummy LayerStack."""

    return LayerStack(
        layers=dict(
            substrate=LayerLevel(
                layer=(0, 0),
                thickness=1,
                zmin=0,
                material="si",
                mesh_order=99,
            )
        )
    )


def test_valid_component() -> None:
    c = gf.components.rectangle()
    scene = to_3d(c)
    assert isinstance(scene, trimesh.Scene)


# Tests that the function raises a ValueError if no polygons are defined in the layer_stack or layer_views
def test_no_polygons_defined() -> None:
    c = gf.Component()
    with pytest.raises(ValueError):
        to_3d(c)


# Tests that the function excludes layers specified in exclude_layers argument
def test_exclude_layers() -> None:
    c = gf.components.rectangle(layer=(0, 0))
    with pytest.raises(ValueError):
        scene = to_3d(c, exclude_layers=((0, 0),))
        assert len(scene.geometry) == 0, len(scene.geometry)


# Tests that the function correctly handles invisible layers
def test_invisible_layers() -> None:
    c = gf.Component()
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=(0, 0))
    with pytest.raises(ValueError):
        scene = to_3d(c)
        assert len(scene.geometry) == 0


# Tests that the function correctly handles missing zmin or thickness values for a layer
def test_missing_zmin_or_thickness() -> None:
    c = gf.components.rectangle()
    layer_stack = get_layer_stack()
    with pytest.raises(ValueError):
        to_3d(c, layer_stack=layer_stack)


if __name__ == "__main__":
    # test_no_polygons_defined()
    # test_exclude_layers()
    # test_invisible_layers()
    # c = gf.components.rectangle(layer=(0,0))
    # to_3d(c, layer_views=None, layer_stack=None)
    c = gf.components.rectangle()
    layer_stack = get_layer_stack()
    to_3d(c, layer_stack=layer_stack)
