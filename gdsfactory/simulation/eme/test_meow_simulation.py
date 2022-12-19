import numpy as np

import gdsfactory as gf
from gdsfactory.simulation.eme import meow_calculation
from gdsfactory.tech import LayerStack, get_layer_stack_generic


def test_meow_defaults():

    c = gf.components.taper_cross_section_linear()
    c.show()

    filtered_layerstack = LayerStack(
        layers={
            k: get_layer_stack_generic().layers[k]
            for k in (
                "slab90",
                "core",
                "box",
                "clad",
            )
        }
    )

    sp = meow_calculation(component=c, layerstack=filtered_layerstack)

    for key in sp.keys():
        if key == "wavelengths":
            continue
        entry1, entry2 = key.split(",")
        port1, mode1 = entry1.split("@")
        port2, mode2 = entry2.split("@")
        if port1 != port2 and mode1 == mode2:
            assert np.abs(sp[key]) ** 2 > 0.7
        else:
            assert np.abs(sp[key]) ** 2 < 0.1


if __name__ == "__main__":

    test_meow_defaults()
