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
    net_component = LAYER_STACK.get_component_with_net_layers(
        original_component, portnames=portnames_to_test, delimiter=delimiter
    )
    layernames_after = set(LAYER_STACK.layers.keys())

    # Check we have two new layers in the LayerStack
    assert len(layernames_after - layernames_before) == 2

    # Check we have one new layer in Component (all metal3 is removed by these operations)
    assert len(net_component.get_layers()) == len(original_component.get_layers()) + 1

    # Check new layer is the same as old layer, apart from layer number and name
    old_layer = LAYER_STACK.layers["metal3"]
    new_layer = LAYER_STACK.layers[f"metal3{delimiter}{portnames_to_test[0]}"]

    for varname in vars(LAYER_STACK.layers["metal3"]):
        if varname == "layer":
            continue
        else:
            assert getattr(old_layer, varname) == getattr(new_layer, varname)


if __name__ == "__main__":
    test_component_with_net_layers()
