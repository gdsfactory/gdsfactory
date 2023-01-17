import numpy as np

from gdsfactory.generic_tech import LAYER_STACK
from gdsfactory.simulation.fem.mode_solver import compute_cross_section_modes
from gdsfactory.technology import LayerStack
from femwell import mode_solver


def common(overwrite=False):
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
        wl=1.55,
        num_modes=4,
        order=1,
        radius=np.inf,
        resolutions=resolutions,
        overwrite=overwrite,
    )
    return lams, basis, xs


def test_compute_cross_section_mode():
    lams, basis, xs = common(overwrite=True)
    assert len(lams) == 4
    # Test cache
    lams2, basis2, xs2 = common(overwrite=False)
    assert lams.all() == lams2.all()
    assert basis.get_dofs().flatten().all() == basis2.get_dofs().flatten().all()
    assert xs.all() == xs2.all()


def test_plot_modes():
    lams, basis, xs = common(overwrite=False)
    mode_solver.plot_mode(
        basis=basis,
        mode=np.real(xs[0]),
        plot_vectors=False,
        colorbar=True,
        title="E",
        direction="y",
    )


if __name__ == "__main__":

    test_compute_cross_section_mode()
    test_plot_modes()
