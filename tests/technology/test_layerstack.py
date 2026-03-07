import pytest

import gdsfactory as gf
from gdsfactory.gpdk import LAYER, LAYER_STACK
from gdsfactory.technology import LayerLevel, LayerStack
from gdsfactory.technology.layer_stack import LogicalLayer

nm = 1e-3


# TODO: fix this test
@pytest.mark.skip(
    reason="Skipping as it is not implemented yet for the new LayerStack."
)
def test_layerstack_to_klayout_3d_script() -> None:
    assert LAYER_STACK.get_klayout_3d_script()


def test_layerstack_filtered() -> None:
    ls2 = LAYER_STACK.filtered(["metal1", "metal2"])
    assert len(ls2.layers) == 2, len(ls2.layers)


def test_layerstack_copy() -> None:
    ls1 = LAYER_STACK
    ls2 = LAYER_STACK.model_copy()
    ls2.layers["metal5"] = ls2.layers["metal1"]
    assert len(ls2.layers) == len(ls1.layers) + 1


def test_layer_level() -> None:
    layers = ["WG", (1, 0), LAYER.WG]

    for layer in layers:
        level = LayerLevel(
            layer=layer,
            thickness=220 * nm,
            thickness_tolerance=5 * nm,
            material="Si",
            mesh_order=2,
            zmin=0,
            sidewall_angle=10,
            sidewall_angle_tolerance=2,
        )
        level_layer = level.layer
        assert isinstance(level_layer, LogicalLayer)
        layer_ = gf.get_layer(level_layer.layer)
        assert isinstance(layer_, int)
        assert int(layer_) == 1, int(layer_)


def test_get_layer_to_mesh_order() -> None:
    ls = LayerStack(
        layers=dict(
            core=LayerLevel(
                layer=LogicalLayer(layer=LAYER.WG),
                thickness=0.22,
                zmin=0,
                mesh_order=1,
            ),
            slab=LayerLevel(  # test that when the info dict contains mesh_order, it takes precedence over the top level mesh_order
                layer=LogicalLayer(layer=LAYER.SLAB90),
                thickness=0.09,
                zmin=0,
                info={"mesh_order": 2},
            ),
            metal=LayerLevel(  # test that if mesh_order is not set at the top level or in info, the default mesh_order is used (3)
                layer=LogicalLayer(layer=LAYER.HEATER),
                thickness=1.0,
                zmin=1.0,
            ),
        )
    )

    result = ls.get_layer_to_mesh_order()
    assert result[LogicalLayer(layer=LAYER.WG)] == 1
    assert result[LogicalLayer(layer=LAYER.SLAB90)] == 2
    assert result[LogicalLayer(layer=LAYER.HEATER)] == 3
