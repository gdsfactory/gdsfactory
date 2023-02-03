from gdsfactory.generic_tech import LAYER_STACK


def test_layerstack():
    assert LAYER_STACK.get_klayout_3d_script()


if __name__ == "__main__":
    test_layerstack()
