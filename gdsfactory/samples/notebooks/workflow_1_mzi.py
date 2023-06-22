# # MZI filter
#
# In this example we will go over a [Mach–Zehnder interferometer](https://en.wikipedia.org/wiki/Mach%E2%80%93Zehnder_interferometer) filter design.
#

# ## Calculations

# +
from typing import Optional

import numpy as np
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()


# +
def mzi(
    wl: np.ndarray,
    neff: Optional[float],
    neff1: Optional[float] = None,
    neff2: Optional[float] = None,
    delta_length: Optional[float] = None,
    length1: Optional[float] = 0,
    length2: Optional[float] = None,
) -> np.ndarray:
    """Returns Frequency Domain Response of an MZI interferometer in linear units.

    Args:
        wl: wavelength in  um.
        neff: effective index.
        neff1: effective index branch 1.
        neff2: effective index branch 2.
        delta_length: length difference L2-L1.
        length1: length of branch 1.
        length2: length of branch 2.
    """
    k_0 = 2 * np.pi / wl
    length2 = length2 or length1 + delta_length
    delta_length = delta_length or np.abs(length2 - length1)
    neff1 = neff1 or neff
    neff2 = neff2 or neff

    E_out = 0.5 * (
        np.exp(1j * k_0 * neff1 * (length1 + delta_length))
        + np.exp(1j * k_0 * neff2 * length1)
    )
    return np.abs(E_out) ** 2


if __name__ == "__main__":
    import gdsfactory.simulation.gtidy3d as gt
    import matplotlib.pyplot as plt

    nm = 1e-3
    strip = gt.modes.Waveguide(
        wavelength=1.55,
        core_width=500 * nm,
        core_thickness=220 * nm,
        slab_thickness=0.0,
        core_material="si",
        clad_material="sio2",
    )

    neff = 2.46  # Effective index of the waveguides
    wl0 = 1.55  # [μm] the wavelength at which neff and ng are defined
    wl = np.linspace(1.5, 1.6, 1000)  # [μm] Wavelengths to sweep over
    ngs = [4.182551, 4.169563, 4.172917]
    thicknesses = [210, 220, 230]

    length = 4e3
    dn = np.pi / length

    polyfit_TE1550SOI_220nm = np.array(
        [
            1.02478963e-09,
            -8.65556534e-08,
            3.32415694e-06,
            -7.68408985e-05,
            1.19282177e-03,
            -1.31366332e-02,
            1.05721429e-01,
            -6.31057637e-01,
            2.80689677e00,
            -9.26867694e00,
            2.24535191e01,
            -3.90664800e01,
            4.71899278e01,
            -3.74726005e01,
            1.77381560e01,
            -1.12666286e00,
        ]
    )

    def neff_w(w):
        return np.poly1d(polyfit_TE1550SOI_220nm)(w)

    w0 = 450 * nm
    dn1 = neff_w(w0 + 1 * nm / 2) - neff_w(w0 - 1 * nm / 2)
    dn5 = neff_w(w0 + 5 * nm / 2) - neff_w(w0 - 5 * nm / 2)
    dn10 = neff_w(w0 + 10 * nm / 2) - neff_w(w0 - 10 * nm / 2)

    pi_length1 = np.pi / dn1
    pi_length5 = np.pi / dn5
    pi_length10 = np.pi / dn10

    print(f"pi_length = {pi_length1:.0f}um for 1nm width variation")
    print(f"pi_length = {pi_length5:.0f}um for 5nm width variation")
    print(f"pi_length = {pi_length10:.0f}um for 10nm width variation")

    dn = dn1
    p = mzi(wl, neff=neff, neff1=neff + dn, neff2=neff + dn, delta_length=10)
    plt.plot(wl, p)
    plt.title("MZI")
    plt.xlabel("wavelength (um)")
    plt.ylabel("Power Transmission")
    plt.grid()
    plt.legend()
    plt.show()
# -

# ## Mode solver
#
# For waveguides you can compute the EM modes.

# +
import numpy as np
import gdsfactory.simulation.gtidy3d as gt
import matplotlib.pyplot as plt

nm = 1e-3
strip = gt.modes.Waveguide(
    wavelength=1.55,
    core_width=0.5,
    core_thickness=0.22,
    slab_thickness=0.0,
    core_material="si",
    clad_material="sio2",
    group_index_step=10 * nm,
)
strip.plot_field(field_name="Ex", mode_index=0)  # TE
# -

ng = strip.n_group[0]
ng

# ## FDTD
#
# Lets compute the Sparameters of a 1x2 power splitter using [tidy3D](https://docs.flexcompute.com/projects/tidy3d/en/latest/), which is a fast GPU based FDTD commercial solver.
#
# To run, you need to [create an account](https://simulation.cloud/) and add credits. The number of credits that each simulation takes depends on the simulation computation time.
#
# ![cloud_model](https://i.imgur.com/5VTCPLR.png)

import gdsfactory.simulation as sim
import gdsfactory as gf
import gdsfactory.simulation.gtidy3d as gt
import gdsfactory.components as pdk
from gdsfactory.config import PATH


c = pdk.mmi1x2()
c.plot()

# +
# sp = gt.write_sparameters(c)
# -

sp = np.load(PATH.sparameters_repo / "mmi1x2_507de731d50770de9096ac9f23321daa.npz")

sim.plot.plot_sparameters(sp)

sim.plot.plot_loss1x2(sp)

# ## Circuit simulation
#
# For the simulations you need to build Sparameters models for your components using FDTD or other methods.
#
# ![demo](https://i.imgur.com/RSOTDIN.png)
#
# Sparameters are common in RF and photonic simulation.
#
# We are going to simulate a MZI interferometer circuit.
# For that we need to simulate each of the component Sparameters in tidy3d and then SAX Sparameter circuit solver to solve the Sparameters for the circuit.
#
# We will be using SAX which is an open source circuit simulator.
#
#
# ```
# set PIP_FIND_LINKS="https://whls.blob.core.windows.net/unstable/index.html"
# pip install sax jax
# pip install "jaxlib[cuda111]" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
# pip install "gdsfactory[tidy3d,sax]"
# ```

mzi10 = gf.components.mzi(splitter=c, delta_length=10)
mzi10

# +
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import sax

import gdsfactory as gf
import gdsfactory.simulation.sax as gsax


# +
def straight(wl=1.5, length=10.0, neff=2.4) -> sax.SDict:
    return sax.reciprocal({("o1", "o2"): jnp.exp(2j * jnp.pi * neff * length / wl)})


def bend_euler(wl=1.5, length=20.0):
    """Assumes a reduced transmission for the euler bend compared to a straight"""
    return {k: 0.99 * v for k, v in straight(wl=wl, length=length).items()}


# -

mmi1x2 = gsax.read.model_from_npz(sp)
models = {
    "bend_euler": bend_euler,
    "mmi1x2": mmi1x2,
    "straight": straight,
}

netlist = mzi10.get_netlist()
circuit, _ = sax.circuit(netlist=netlist, models=models)

wl = np.linspace(1.5, 1.6)
S = circuit(wl=wl)
plt.figure(figsize=(14, 4))
plt.title("MZI")
plt.plot(1e3 * wl, 10 * np.log10(jnp.abs(S["o1", "o2"]) ** 2))
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.grid(True)
plt.show()

mzi20 = gf.components.mzi(splitter=c, delta_length=20)
mzi20

netlist = mzi20.get_netlist()
circuit, _ = sax.circuit(netlist=netlist, models=models)

wl = np.linspace(1.5, 1.6)
S = circuit(wl=wl)
plt.figure(figsize=(14, 4))
plt.title("MZI")
plt.plot(1e3 * wl, 10 * np.log10(jnp.abs(S["o1", "o2"]) ** 2))
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.grid(True)
plt.show()
