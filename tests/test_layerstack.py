from gdsfactory.generic_tech import LAYER_STACK


def test_layerstack() -> None:
    assert LAYER_STACK.get_klayout_3d_script()


def test_layerstack_filtered() -> None:
    ls2 = LAYER_STACK.filtered(["metal1", "metal2"])
    assert len(ls2.layers) == 2, len(ls2.layers)


if __name__ == "__main__":
    test_layerstack_filtered()
    test_layerstack()
