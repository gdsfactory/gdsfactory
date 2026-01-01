import pytest
import trimesh

import gdsfactory as gf
from gdsfactory.export.to_3d import to_3d
from gdsfactory.gpdk.layer_map import LAYER
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer


def get_layer_stack() -> LayerStack:
    """Returns dummy LayerStack."""
    return LayerStack(
        layers=dict(
            substrate=LayerLevel(
                layer=LogicalLayer(layer=(1, 0)),
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


def test_no_polygons_defined() -> None:
    """Tests that the function raises a ValueError if no polygons are defined in the layer_stack or layer_views."""
    c = gf.Component()
    with pytest.raises(ValueError):
        to_3d(c)


def test_exclude_layers() -> None:
    """Tests that the function correctly excludes layers specified in exclude_layers argument."""
    c = gf.components.rectangle(layer=LAYER.WG)
    with pytest.raises(ValueError):
        scene = to_3d(c, exclude_layers=(LAYER.WG,))
        assert len(scene.geometry) == 0, len(scene.geometry)


def test_invisible_layers() -> None:
    """Tests that the function correctly handles invisible layers."""
    c = gf.Component()
    c.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=(2, 0))
    with pytest.raises(ValueError):
        scene = to_3d(c, layer_stack=get_layer_stack())
        assert len(scene.geometry) == 0, len(scene.geometry)


def test_missing_zmin_or_thickness() -> None:
    """Tests that the function correctly handles missing zmin or thickness values for a layer."""
    c = gf.components.rectangle(layer=LAYER.SLAB90)
    layer_stack = get_layer_stack()
    with pytest.raises(ValueError):
        to_3d(c, layer_stack=layer_stack)
