import numpy as np

import gdsfactory as gf
from gdsfactory.generic_tech import LAYER_STACK
from gdsfactory.simulation.fem.mode_solver import compute_cross_section_modes, Modes
from gdsfactory.technology import LayerStack

NUM_MODES = 1

PDK = gf.get_generic_pdk()
PDK.activate()


def compute_modes(
    overwrite: bool = True, with_cache: bool = False, num_modes: int = NUM_MODES
) -> Modes:
    filtered_layerstack = LayerStack(
        layers={
            k: LAYER_STACK.layers[k]
            for k in (
                "core",
                "clad",
                "slab90",
                "box",
            )
        }
    )

    filtered_layerstack.layers["core"].thickness = 0.2

    resolutions = {
        "core": {"resolution": 0.02, "distance": 2},
        "clad": {"resolution": 0.2, "distance": 1},
        "box": {"resolution": 0.2, "distance": 1},
        "slab90": {"resolution": 0.05, "distance": 1},
    }
    return compute_cross_section_modes(
        cross_section="rib",
        layerstack=filtered_layerstack,
        wavelength=1.55,
        num_modes=num_modes,
        order=1,
        radius=np.inf,
        resolutions=resolutions,
        overwrite=overwrite,
        with_cache=with_cache,
    )


def test_compute_cross_section_mode() -> None:
    modes = compute_modes()
    assert len(modes) == NUM_MODES, len(modes)


if __name__ == "__main__":
    test_compute_cross_section_mode()
