from gdsfactory.generic_tech import LAYER_STACK


def test_layerstack() -> None:
    assert LAYER_STACK.get_klayout_3d_script()


def test_layerstack_filtered() -> None:
    ls2 = LAYER_STACK.filtered(["xs_m1", "xs_m2"])
    assert len(ls2.layers) == 2, len(ls2.layers)


def test_layerstack_copy() -> None:
    ls1 = LAYER_STACK
    ls2 = LAYER_STACK.model_copy()
    ls2.layers["metal5"] = ls2.layers["xs_m2"]
    assert len(ls2.layers) == len(ls1.layers) + 1


if __name__ == "__main__":
    # test_layerstack_filtered()
    # test_layerstack()
    test_layerstack_copy()
