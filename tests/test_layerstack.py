from gdsfactory.components import straight_heater_metal
from gdsfactory.generic_tech import LAYER_STACK


def test_layerstack() -> None:
    assert LAYER_STACK.get_klayout_3d_script()


def test_component_with_net_layers():
    # Hardcoded settings for now
    delimiter = "#"
    portnames_to_test = ["r_e2", "l_e4"]
    layernames_before = set(LAYER_STACK.layers.keys())
    original_component = straight_heater_metal()

    # Run the function
    LAYER_STACK.get_component_with_net_layers(
        original_component,
        portnames=portnames_to_test,
        delimiter=delimiter,
        remove_empty_layers=False,
    )
    layernames_after = set(LAYER_STACK.layers.keys())

    # Check we have two new layers in the LayerStack
    assert len(layernames_after - layernames_before) == 2

    # Check new layer is the same as old layer, apart from layer number and name
    old_layer = LAYER_STACK.layers["metal3"]
    new_layer = LAYER_STACK.layers[f"metal3{delimiter}{portnames_to_test[0]}"]

    for varname in vars(LAYER_STACK.layers["metal3"]):
        if varname == "layer":
            continue
        else:
            assert getattr(old_layer, varname) == getattr(new_layer, varname)

    # Test remove old layers
    LAYER_STACK.get_component_with_net_layers(
        original_component,
        portnames=portnames_to_test,
        delimiter=delimiter,
        remove_empty_layers=True,
    )
    # Assert that "metal3" does not exist in the layers
    assert "metal3" not in LAYER_STACK.layers


def test_layerstack_filtered() -> None:
    ls2 = LAYER_STACK.filtered(["metal1", "metal2"])
    assert len(ls2.layers) == 2


if __name__ == "__main__":
    # test_layerstack_filtered()
    test_component_with_net_layers()
