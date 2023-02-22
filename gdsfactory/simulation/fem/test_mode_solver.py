import numpy as np

import gdsfactory as gf
from gdsfactory.generic_tech import LAYER_STACK
from gdsfactory.simulation.fem.mode_solver import compute_cross_section_modes
from gdsfactory.technology import LayerStack
from femwell import mode_solver


NUM_MODES = 1

PDK = gf.get_generic_pdk()
PDK.activate()


def compute_modes(
    overwrite: bool = True, with_cache: bool = False, num_modes: int = NUM_MODES
):
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
    lams, basis, xs = compute_cross_section_modes(
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
    return lams, basis, xs


def test_compute_cross_section_mode():
    lams, basis, xs = compute_modes()
    assert len(lams) == NUM_MODES


def test_compute_cross_section_mode_cache():
    # write mode in cache
    lams, basis, xs = compute_modes(with_cache=True, overwrite=False)

    # load mode from cache
    lams, basis, xs = compute_modes(with_cache=True, overwrite=False)
    mode_solver.plot_mode(
        basis=basis,
        mode=np.real(xs[0]),
        plot_vectors=False,
        colorbar=True,
        title="E",
        direction="y",
    )


if __name__ == "__main__":
    test_compute_cross_section_mode_cache()
    # lams, basis, xs = compute_modes(with_cache=True, overwrite=False)
    # mode_solver.plot_mode(
    #     basis=basis,
    #     mode=np.real(xs[0]),
    #     plot_vectors=False,
    #     colorbar=True,
    #     title="E",
    #     direction="y",
    # )
