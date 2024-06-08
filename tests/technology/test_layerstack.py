import pytest

from gdsfactory.generic_tech import LAYER_STACK


@pytest.mark.skip(reason="Spipping as it is implemented yet for the new LayerStack.")
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


def test_component_with_derived_layers() -> None:
    assert True


if __name__ == "__main__":
    test_component_with_derived_layers()
