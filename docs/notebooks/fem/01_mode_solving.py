# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Finite-element mode solver
#
# You can mesh any component cross-section and solve the PDEs thanks to [femwell](https://helgegehring.github.io/femwell) mode - solver.
#
# Unlike other mode solvers, this actually uses the component geometry instead of a hardcoded geometry.
#
# You can directly compute the modes of a Gdsfactory cross-section (internally, it defines a "uz" mesh  perpendicular to a straight component with the provided cross-section).
#
# You can also downsample layers from the LayerStack, and modify both the cross-section and LayerStack  prior to simulation to change the geometry. You can also define refractive indices on the active PDK.

# +
import matplotlib.pyplot as plt
import gdsfactory as gf
from tqdm.auto import tqdm
import numpy as np
from femwell import mode_solver
from gdsfactory.simulation.fem.mode_solver import compute_cross_section_modes
from gdsfactory.technology import LayerStack
from gdsfactory.cross_section import rib
from gdsfactory.generic_tech import LAYER_STACK

import sys
import logging
from rich.logging import RichHandler
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

logger = logging.getLogger()
logger.removeHandler(sys.stderr)
logging.basicConfig(level="WARNING", datefmt="[%X]", handlers=[RichHandler()])

# +
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

filtered_layerstack.layers[
    "core"
].thickness = 0.22  # Perturb the layerstack before simulating

filtered_layerstack.layers[
    "slab90"
].thickness = 0.09  # Perturb the layerstack before simulating

resolutions = {}
resolutions["core"] = {"resolution": 0.02, "distance": 2}
resolutions["clad"] = {"resolution": 0.2, "distance": 1}
resolutions["box"] = {"resolution": 0.2, "distance": 1}
resolutions["slab90"] = {"resolution": 0.05, "distance": 1}

lams, basis, xs = compute_cross_section_modes(
    cross_section=rib(width=0.6),
    layerstack=filtered_layerstack,
    wavelength=1.55,
    num_modes=4,
    resolutions=resolutions,
)
# -

# The solver returns the effective indices (lams), FEM basis functions (basis) and eigenvectors (xs):

lams, basis, xs

# You can use them as inputs to other [femwell mode solver functions](https://github.com/HelgeGehring/femwell/blob/main/femwell/mode_solver.py) to inspect or analyze the modes:

fig = mode_solver.plot_mode(
    basis, np.real(xs[0]), plot_vectors=False, colorbar=True, title="E", direction="y"
)

fig[0]

te_frac = mode_solver.calculate_te_frac(basis, xs[0])
te_frac

# ## Sweep waveguide width

# +
widths = np.linspace(0.2, 2, 20)
num_modes = 4
all_lams = np.zeros((widths.shape[0], num_modes))
all_te_fracs = np.zeros((widths.shape[0], num_modes))

for i, width in enumerate(tqdm(widths)):
    lams, basis, xs = compute_cross_section_modes(
        cross_section=gf.cross_section.strip(width=width),
        layerstack=filtered_layerstack,
        wavelength=1.55,
        num_modes=num_modes,
        resolutions=resolutions,
        wafer_padding=2,
    )
    all_lams[i] = lams
    all_te_fracs[i, :] = [
        mode_solver.calculate_te_frac(basis, xs[idx]) for idx in range(num_modes)
    ]

# +
all_lams = np.real(all_lams)
plt.xlabel("waveguide Width [µm]")
plt.ylabel("Effective refractive index")
plt.ylim(1.444, np.max(all_lams) + 0.1 * (np.max(all_lams) - 1.444))

for lams, te_fracs in zip(all_lams.T, all_te_fracs.T):
    plt.plot(widths, lams)
    plt.scatter(widths, lams, c=te_fracs, cmap="cool")
