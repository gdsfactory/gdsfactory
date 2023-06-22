# # SAX circuit simulator
#
# [SAX](https://flaport.github.io/sax/) is a circuit solver written in JAX, writing your component models in SAX enables you not only to get the function values but the gradients, this is useful for circuit optimization.
#
# This tutorial has been adapted from the SAX Quick Start.
#
# You can install sax with pip (read the SAX install instructions [here](https://github.com/flaport/sax#installation))
#
# ```
# # ! pip install sax
# ```

# +
from tqdm import trange
from tqdm.notebook import trange

from numpy.fft import fft2, fftfreq, fftshift, ifft2
from typing import List
from functools import partial
import sys
from pprint import pprint
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import constants
import jax.example_libraries.optimizers as opt
import jax.numpy as jnp
import jax

import gdsfactory.simulation.gtidy3d as gt
import gdsfactory as gf
import gdsfactory.simulation.sax as gs

import matplotlib.pyplot as plt
import sax

import sys
import logging
from rich.logging import RichHandler
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

logger = logging.getLogger()
logger.removeHandler(sys.stderr)
logging.basicConfig(level="WARNING", datefmt="[%X]", handlers=[RichHandler()])

gf.config.set_plot_options(show_subports=False)
# -

# ## Scatter *dictionaries*
#
# The core datastructure for specifying scatter parameters in SAX is a dictionary... more specifically a dictionary which maps a port combination (2-tuple) to a scatter parameter (or an array of scatter parameters when considering multiple wavelengths for example). Such a specific dictionary mapping is called ann `SDict` in SAX (`SDict ≈ Dict[Tuple[str,str], float]`).
#
# Dictionaries are in fact much better suited for characterizing S-parameters than, say, (jax-)numpy arrays due to the inherent sparse nature of scatter parameters. Moreover, dictionaries allow for string indexing, which makes them much more pleasant to use in this context.
#
# ```
# o2            o3
#    \        /
#     ========
#    /        \
# o1            o4
# ```

coupling = 0.5
kappa = coupling**0.5
tau = (1 - coupling) ** 0.5
coupler_dict = {
    ("o1", "o4"): tau,
    ("o4", "o1"): tau,
    ("o1", "o3"): 1j * kappa,
    ("o3", "o1"): 1j * kappa,
    ("o2", "o4"): 1j * kappa,
    ("o4", "o2"): 1j * kappa,
    ("o2", "o3"): tau,
    ("o3", "o2"): tau,
}
coupler_dict

#  it can still be tedious to specify every port in the circuit manually. SAX therefore offers the `reciprocal` function, which auto-fills the reverse connection if the forward connection exist. For example:

# +
coupler_dict = sax.reciprocal(
    {
        ("o1", "o4"): tau,
        ("o1", "o3"): 1j * kappa,
        ("o2", "o4"): 1j * kappa,
        ("o2", "o3"): tau,
    }
)

coupler_dict
# -

# ## Parametrized Models
#
# Constructing such an `SDict` is easy, however, usually we're more interested in having parametrized models for our components. To parametrize the coupler `SDict`, just wrap it in a function to obtain a SAX `Model`, which is a keyword-only function mapping to an `SDict`:
#


# +
def coupler(coupling=0.5) -> sax.SDict:
    kappa = coupling**0.5
    tau = (1 - coupling) ** 0.5
    return sax.reciprocal(
        {
            ("o1", "o4"): tau,
            ("o1", "o3"): 1j * kappa,
            ("o2", "o4"): 1j * kappa,
            ("o2", "o3"): tau,
        }
    )


coupler(coupling=0.3)


# -


def waveguide(wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss=0.0) -> sax.SDict:
    dwl = wl - wl0
    dneff_dwl = (ng - neff) / wl0
    neff = neff - dwl * dneff_dwl
    phase = 2 * jnp.pi * neff * length / wl
    transmission = 10 ** (-loss * length / 20) * jnp.exp(1j * phase)
    return sax.reciprocal(
        {
            ("o1", "o2"): transmission,
        }
    )


# ### Waveguide model
#
# You can create a dispersive waveguide model in SAX.

# Lets compute the effective index `neff` and group index `ng` for a 1550nm 500nm straight waveguide

nm = 1e-3
strip = gt.modes.Waveguide(
    wavelength=1.55,
    core_width=500 * nm,
    core_thickness=220 * nm,
    slab_thickness=0.0,
    core_material="si",
    clad_material="sio2",
    group_index_step=10 * nm,
)
strip.plot_field(field_name="Ex", mode_index=0)  # TE

neff = strip.n_eff[0]
neff

ng = strip.n_group[0]
ng

straight_sc = partial(gs.models.straight, neff=neff, ng=ng)

gs.plot_model(straight_sc)
plt.ylim(-1, 1)

gs.plot_model(straight_sc, phase=True)

# ### Coupler model

c = gf.components.coupler(length=10, gap=0.2)
c.plot()

nm = 1e-3
cp = gt.modes.WaveguideCoupler(
    wavelength=1.55,
    core_width=(500 * nm, 500 * nm),
    gap=200 * nm,
    core_thickness=220 * nm,
    slab_thickness=0 * nm,
    core_material="si",
    clad_material="sio2",
)
cp.plot_field(field_name="Ex", mode_index=0)  # even mode

cp.plot_field(field_name="Ex", mode_index=1)  # odd mode

# +
coupler = gt.modes.WaveguideCoupler(
    wavelength=1.55,
    core_width=(0.45, 0.45),
    core_thickness=220 * nm,
    core_material="si",
    clad_material="sio2",
    num_modes=4,
    gap=100 * nm,
)

print("\nCoupler:", coupler)
print("Effective indices:", coupler.n_eff)
print("Mode areas:", coupler.mode_area)
print("Coupling length:", coupler.coupling_length())

gaps = np.linspace(0.05, 0.15, 11)
lengths = gt.modes.sweep_coupling_length(coupler, gaps)
plt.plot(gaps, lengths)
plt.xlabel("Gap (μm)")
plt.ylabel("Coupling length (μm)")
# -

# For a 200nm gap the effective index difference `dn` is `0.026`, which means that there is 100% power coupling over 29.4

coupler_sc = partial(gs.models.coupler, dn=0.026, length=0, coupling0=0)
gs.plot_model(coupler_sc)

# If we ignore the coupling from the bend `coupling0 = 0` we know that for a 3dB coupling we need half of the `lc` length, which is the length needed to coupler `100%` of power.

coupler_sc = partial(gs.models.coupler, dn=0.026, length=29.4 / 2, coupling0=0)
gs.plot_model(coupler_sc)

# ### FDTD Sparameters model
#
# You can also fit a model from Sparameter FDTD simulation data from tidy3d, Lumerical or MEEP.

# ## Model fit
#
# You can fit a sax model to Sparameter FDTD simulation data.

filepath = gf.config.PATH.test_data / "sp" / "coupler_G224n_L20_S220.csv"

coupler_fdtd = gs.read.model_from_csv(
    filepath=filepath,
    xkey="wavelength_nm",
    prefix="S",
    xunits=1e-3,
)

gs.plot_model(coupler_fdtd)

# Lets fit the coupler spectrum with a linear regression `sklearn` fit

# +
f = jnp.linspace(constants.c / 1.0e-6, constants.c / 2.0e-6, 500) * 1e-12  # THz
wl = constants.c / (f * 1e12) * 1e6  # um

coupler_fdtd = gs.read.model_from_csv(
    filepath, xkey="wavelength_nm", prefix="S", xunits=1e-3
)
sd = coupler_fdtd(wl=wl)

k = sd["o1", "o3"]
t = sd["o1", "o4"]
s = t + k
a = t - k
# -

# Lets fit the symmetric (t+k) and antisymmetric (t-k) transmission
#
# ### Symmetric

plt.plot(wl, jnp.abs(s))
plt.grid(True)
plt.xlabel("Frequency [THz]")
plt.ylabel("Transmission")
plt.title("symmetric (transmission + coupling)")
plt.legend()
plt.show()

plt.plot(wl, jnp.abs(a))
plt.grid(True)
plt.xlabel("Frequency [THz]")
plt.ylabel("Transmission")
plt.title("anti-symmetric (transmission - coupling)")
plt.legend()
plt.show()

# +
r = LinearRegression()


def fX(x, _order=8):
    return x[:, None] ** (
        jnp.arange(_order)[None, :]
    )  # artificially create more 'features' (wl**2, wl**3, wl**4, ...)


X = fX(wl)
r.fit(X, jnp.abs(s))
asm, bsm = r.coef_, r.intercept_


def fsm(x):
    return fX(x) @ asm + bsm  # fit symmetric module fiir


plt.plot(wl, jnp.abs(s), label="data")
plt.plot(wl, fsm(wl), label="fit")
plt.grid(True)
plt.xlabel("Frequency [THz]")
plt.ylabel("Transmission")
plt.legend()
plt.show()

# +
r = LinearRegression()
r.fit(X, jnp.unwrap(jnp.angle(s)))
asp, bsp = r.coef_, r.intercept_


def fsp(x):
    return fX(x) @ asp + bsp  # fit symmetric phase


plt.plot(wl, jnp.unwrap(jnp.angle(s)), label="data")
plt.plot(wl, fsp(wl), label="fit")
plt.grid(True)
plt.xlabel("Frequency [THz]")
plt.ylabel("Angle [rad]")
plt.legend()
plt.show()


# -


def fs(x):
    return fsm(x) * jnp.exp(1j * fsp(x))


# Lets fit the symmetric (t+k) and antisymmetric (t-k) transmission
#
# ### Anti-Symmetric

# +
r = LinearRegression()
r.fit(X, jnp.abs(a))
aam, bam = r.coef_, r.intercept_


def fam(x):
    return fX(x) @ aam + bam


plt.plot(wl, jnp.abs(a))
plt.plot(wl, fam(wl))
plt.grid(True)
plt.xlabel("Frequency [THz]")
plt.ylabel("Transmission")
plt.legend()
plt.show()

# +
r = LinearRegression()
r.fit(X, jnp.unwrap(jnp.angle(a)))
aap, bap = r.coef_, r.intercept_


def fap(x):
    return fX(x) @ aap + bap


plt.plot(wl, jnp.unwrap(jnp.angle(a)))
plt.plot(wl, fap(wl))
plt.grid(True)
plt.xlabel("Frequency [THz]")
plt.ylabel("Angle [rad]")
plt.legend()
plt.show()


# -


def fa(x):
    return fam(x) * jnp.exp(1j * fap(x))


# ### Total

# +
t_ = 0.5 * (fs(wl) + fa(wl))

plt.plot(wl, jnp.abs(t))
plt.plot(wl, jnp.abs(t_))
plt.xlabel("Frequency [THz]")
plt.ylabel("Transmission")

# +
k_ = 0.5 * (fs(wl) - fa(wl))

plt.plot(wl, jnp.abs(k))
plt.plot(wl, jnp.abs(k_))
plt.xlabel("Frequency [THz]")
plt.ylabel("Coupling")


# -


@jax.jit
def coupler(wl=1.5):
    wl = jnp.asarray(wl)
    wl_shape = wl.shape
    wl = wl.ravel()
    t = (0.5 * (fs(wl) + fa(wl))).reshape(*wl_shape)
    k = (0.5 * (fs(wl) - fa(wl))).reshape(*wl_shape)
    sdict = {
        ("o1", "o4"): t,
        ("o1", "o3"): k,
        ("o2", "o3"): k,
        ("o2", "o4"): t,
    }
    return sax.reciprocal(sdict)


# +
f = jnp.linspace(constants.c / 1.0e-6, constants.c / 2.0e-6, 500) * 1e-12  # THz
wl = constants.c / (f * 1e12) * 1e6  # um

coupler_fdtd = gs.read.model_from_csv(
    filepath, xkey="wavelength_nm", prefix="S", xunits=1e-3
)
sd = coupler_fdtd(wl=wl)
sd_ = coupler(wl=wl)

T = jnp.abs(sd["o1", "o4"]) ** 2
K = jnp.abs(sd["o1", "o3"]) ** 2
T_ = jnp.abs(sd_["o1", "o4"]) ** 2
K_ = jnp.abs(sd_["o1", "o3"]) ** 2
dP = jnp.unwrap(jnp.angle(sd["o1", "o3"]) - jnp.angle(sd["o1", "o4"]))
dP_ = jnp.unwrap(jnp.angle(sd_["o1", "o3"]) - jnp.angle(sd_["o1", "o4"]))

plt.figure(figsize=(12, 3))
plt.plot(wl, T, label="T (fdtd)", c="C0", ls=":", lw="6")
plt.plot(wl, T_, label="T (model)", c="C0")

plt.plot(wl, K, label="K (fdtd)", c="C1", ls=":", lw="6")
plt.plot(wl, K_, label="K (model)", c="C1")

plt.ylim(-0.05, 1.05)
plt.grid(True)

plt.twinx()
plt.plot(wl, dP, label="ΔΦ (fdtd)", color="C2", ls=":", lw="6")
plt.plot(wl, dP_, label="ΔΦ (model)", color="C2")

plt.xlabel("Frequency [THz]")
plt.ylabel("Transmission")
plt.figlegend(bbox_to_anchor=(1.08, 0.9))
plt.show()
# -

# ## SAX gdsfactory Compatibility
# > From Layout to Circuit Model
#
# If you define your SAX S parameter models for your components, you can directly simulate your circuits from gdsfactory

mzi = gf.components.mzi(delta_length=10)
mzi

mzi.plot_netlist()

netlist = mzi.get_netlist()
pprint(netlist["connections"])


# The netlist has three different components:
#
# 1. straight
# 2. mmi1x2
# 3. bend_euler
#
# You need models for each subcomponents to simulate the Component.
#


# +
def straight(wl=1.5, length=10.0, neff=2.4) -> sax.SDict:
    return sax.reciprocal({("o1", "o2"): jnp.exp(2j * jnp.pi * neff * length / wl)})


def mmi1x2():
    """Assumes a perfect 1x2 splitter"""
    return sax.reciprocal(
        {
            ("o1", "o2"): 0.5**0.5,
            ("o1", "o3"): 0.5**0.5,
        }
    )


def bend_euler(wl=1.5, length=20.0):
    """ "Let's assume a reduced transmission for the euler bend compared to a straight"""
    return {k: 0.99 * v for k, v in straight(wl=wl, length=length).items()}


models = {
    "bend_euler": bend_euler,
    "mmi1x2": mmi1x2,
    "straight": straight,
}
# -

circuit, _ = sax.circuit(netlist=netlist, models=models)

# +
wl = np.linspace(1.5, 1.6)
S = circuit(wl=wl)

plt.figure(figsize=(14, 4))
plt.title("MZI")
plt.plot(1e3 * wl, jnp.abs(S["o1", "o2"]) ** 2)
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.grid(True)
plt.show()
# -

mzi = gf.components.mzi(delta_length=20)  # Double the length, reduces FSR by 1/2
mzi

# +
circuit, _ = sax.circuit(netlist=mzi.get_netlist(), models=models)

wl = np.linspace(1.5, 1.6, 256)
S = circuit(wl=wl)

plt.figure(figsize=(14, 4))
plt.title("MZI")
plt.plot(1e3 * wl, jnp.abs(S["o1", "o2"]) ** 2)
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.grid(True)
plt.show()
# -

# ## Layout aware Monte Carlo
#
# You can model the manufacturing variations on the performance of photonics thanks to the fast SAX circuit simulator with layout information and wafer maps of waveguide width and layer thickness variations.
#
# The width and height variations can be extracted from:
#
# - Ring resonators [2017](https://opg.optica.org/oe/fulltext.cfm?uri=oe-25-9-9712&id=363202)
# - MZI interferometers [2019](https://ieeexplore.ieee.org/abstract/document/8675367)

# ### Waveguide Model
#
# To improve the waveguide model you need to find the effective index of the waveguide in relation to its parameters (width and thickness) using an open source mode solver.

# +
nm = 1e-3
wavelengths = np.linspace(1.5, 1.6, 10)
widths = np.linspace(400 * nm, 600 * nm, 5)

neffs = gt.modes.sweep_n_eff(
    gt.modes.Waveguide(
        wavelength=wavelengths,
        core_width=widths[0],
        num_modes=1,
        core_thickness=220 * nm,
        slab_thickness=0.0,
        core_material="si",
        clad_material="sio2",
    ),
    core_width=widths,
)

neffs = neffs.values.real
# -

plt.pcolormesh(wavelengths, widths, neffs)
plt.xlabel("λ [μm]")
plt.ylabel("width [μm]")
plt.colorbar()
plt.show()

# +
_grid = [jnp.sort(jnp.unique(widths)), jnp.sort(jnp.unique(wavelengths))]
_data = jnp.asarray(neffs)


@jax.jit
def _get_coordinate(arr1d: jnp.ndarray, value: jnp.ndarray):
    return jnp.interp(value, arr1d, jnp.arange(arr1d.shape[0]))


@jax.jit
def _get_coordinates(arrs1d: List[jnp.ndarray], values: jnp.ndarray):
    # don't use vmap as arrays in arrs1d could have different shapes...
    return jnp.array([_get_coordinate(a, v) for a, v in zip(arrs1d, values)])


@jax.jit
def neff(wl=1.55, width=0.5):
    params = jnp.stack(jnp.broadcast_arrays(jnp.asarray(width), jnp.asarray(wl), 0))
    coords = _get_coordinates(_grid, params)
    return jax.scipy.ndimage.map_coordinates(_data, coords, 1, mode="nearest")


neff(wl=[1.52, 1.58], width=[0.5, 0.55])
# -

wavelengths_ = np.linspace(wavelengths.min(), wavelengths.max(), 100)
widths_ = np.linspace(widths.min(), widths.max(), 100)
wavelengths_, widths_ = np.meshgrid(wavelengths_, widths_)
neffs_ = neff(wavelengths_, widths_)
plt.pcolormesh(wavelengths_, widths_, neffs_)
plt.xlabel("λ [μm]")
plt.ylabel("width [μm]")
plt.colorbar()
plt.show()


# +
def straight(wl=1.55, length=10.0, width=0.5):
    S = {
        ("o1", "o2"): jnp.exp(2j * np.pi * neff(wl=wl, width=width) / wl * length),
    }
    return sax.reciprocal(S)


def mmi1x2():
    """Assumes a perfect 1x2 splitter"""
    return sax.reciprocal(
        {
            ("o1", "o2"): 0.5**0.5,
            ("o1", "o3"): 0.5**0.5,
        }
    )


def mmi2x2():
    S = {
        ("o1", "o3"): 0.5**0.5,
        ("o1", "o4"): 1j * 0.5**0.5,
        ("o2", "o3"): 1j * 0.5**0.5,
        ("o2", "o4"): 0.5**0.5,
    }
    return sax.reciprocal(S)


def bend_euler(wl=1.5, length=20.0, width=0.5):
    """ "Let's assume a reduced transmission for the euler bend compared to a straight"""
    return {k: 0.99 * v for k, v in straight(wl=wl, length=length, width=width).items()}


models = {
    "bend_euler": bend_euler,
    "mmi1x2": mmi1x2,
    "mmi2x2": mmi2x2,
    "straight": straight,
}
# -

# Even though this still is lossless transmission, we're at least modeling the phase correctly.

straight()

circuit, _ = sax.circuit(mzi.get_netlist(), models=models)
circuit()

wl = jnp.linspace(1.51, 1.59, 1000)
S = circuit(wl=wl)
plt.plot(wl, abs(S["o1", "o2"]) ** 2)
plt.ylim(-0.05, 1.05)
plt.xlabel("λ [μm]")
plt.ylabel("T")
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.show()


# ### Circuit model with variability
#
# Let's assume the waveguide width changes with a certain correlation length.
# We can create a 'wafermap' of width variations by randomly varying the width and low pass filtering with a spatial frequency being the inverse of the correlation length.
# There are probably better ways to do this, but this works for this tutorial.
#


def create_wafermaps(placements, correlation_length=1.0, num_maps=1, mean=0.0, std=1.0):
    dx = dy = correlation_length / 200
    xs, ys = [p["x"] for p in placements.values()], [
        p["y"] for p in placements.values()
    ]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    wx, wy = xmax - xmin, ymax - ymin
    xmin, xmax, ymin, ymax = xmin - wx, xmax + wx, ymin - wy, ymax + wy
    x, y = np.arange(xmin, xmax + dx, dx), np.arange(ymin, ymax + dy, dy)
    W0 = np.random.randn(num_maps, x.shape[0], y.shape[0])

    fx, fy = fftshift(fftfreq(x.shape[0], d=x[1] - x[0])), fftshift(
        fftfreq(y.shape[0], d=y[1] - y[0])
    )
    fY, fX = np.meshgrid(fy, fx)
    fW = fftshift(fft2(W0))

    if correlation_length >= min(x.shape[0], y.shape[0]):
        fW = np.zeros_like(fW)
    else:
        fW = np.where(np.sqrt(fX**2 + fY**2)[None] > 1 / correlation_length, 0, fW)

    W = np.abs(fftshift(ifft2(fW))) ** 2
    mean_ = W.mean(1, keepdims=True).mean(2, keepdims=True)
    std_ = W.std(1, keepdims=True).std(2, keepdims=True)
    if (std_ == 0).all():
        std_ = 1

    W = (W - mean_) / std_
    W = W * std + mean
    return x, y, W


# +
placements = mzi.get_netlist()["placements"]
xm, ym, wmaps = create_wafermaps(
    placements, correlation_length=100, mean=0.5, std=0.002, num_maps=100
)

for i, wmap in enumerate(wmaps):
    plt.imshow(wmap, cmap="RdBu")
    plt.show()
    if i == 2:
        break


# -


def widths(xw, yw, wmaps, x, y):
    _wmap_grid = [xw, yw]
    params = jnp.stack(jnp.broadcast_arrays(jnp.asarray(x), jnp.asarray(y)), 0)
    coords = _get_coordinates(_wmap_grid, params)

    map_coordinates = partial(
        jax.scipy.ndimage.map_coordinates, coordinates=coords, order=1, mode="nearest"
    )
    return jax.vmap(map_coordinates)(wmaps)


# Let's now sample the MZI width variation on the wafer map (let's assume a single width variation per point):
#
#
# ### Simple MZI


# +
@gf.cell
def simple_mzi():
    global bend_top1_
    c = gf.Component()

    # instances
    mmi_in = gf.components.mmi1x2()
    mmi_out = gf.components.mmi2x2()
    bend = gf.components.bend_euler()
    half_delay_straight = gf.components.straight(length=10.0)

    # references (sax convention: vars ending in underscore are references)
    mmi_in_ = c << mmi_in
    mmi_out_ = c << mmi_out
    straight_top1_ = c << half_delay_straight
    straight_top2_ = c << half_delay_straight
    bend_top1_ = c << bend
    bend_top2_ = (c << bend).mirror()
    bend_top3_ = (c << bend).mirror()
    bend_top4_ = c << bend
    bend_btm1_ = (c << bend).mirror()
    bend_btm2_ = c << bend
    bend_btm3_ = c << bend
    bend_btm4_ = (c << bend).mirror()

    # connections
    bend_top1_.connect("o1", mmi_in_.ports["o2"])
    straight_top1_.connect("o1", bend_top1_.ports["o2"])
    bend_top2_.connect("o1", straight_top1_.ports["o2"])
    bend_top3_.connect("o1", bend_top2_.ports["o2"])
    straight_top2_.connect("o1", bend_top3_.ports["o2"])
    bend_top4_.connect("o1", straight_top2_.ports["o2"])

    bend_btm1_.connect("o1", mmi_in_.ports["o3"])
    bend_btm2_.connect("o1", bend_btm1_.ports["o2"])
    bend_btm3_.connect("o1", bend_btm2_.ports["o2"])
    bend_btm4_.connect("o1", bend_btm3_.ports["o2"])

    mmi_out_.connect("o1", bend_btm4_.ports["o2"])

    # ports
    c.add_port(
        "o1",
        port=mmi_in_.ports["o1"],
    )
    c.add_port("o2", port=mmi_out_.ports["o3"])
    c.add_port("o3", port=mmi_out_.ports["o4"])
    return c


mzi = simple_mzi()
mzi
# -

circuit, _ = sax.circuit(mzi.get_netlist(), models=models)

# +
mzi_params = sax.get_settings(circuit)
placements = mzi.get_netlist()["placements"]
width_params = {
    k: {"width": widths(xm, ym, wmaps, v["x"], v["y"])}
    for k, v in placements.items()
    if "width" in mzi_params[k]
}

S0 = circuit(wl=wl)
S = circuit(
    wl=wl[:, None],
    **width_params,
)
ps = plt.plot(wl * 1e3, abs(S["o1", "o2"]) ** 2, color="C0", lw=1, alpha=0.1)
nps = plt.plot(wl * 1e3, abs(S0["o1", "o2"]) ** 2, color="C1", lw=2, alpha=1)
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.plot([1550, 1550], [-1, 2], color="black", ls=":")
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.figlegend([*ps[-1:], *nps], ["MC", "nominal"], bbox_to_anchor=(1.1, 0.9))
rmse = jnp.mean(
    jnp.abs(jnp.abs(S["o1", "o2"]) ** 2 - jnp.abs(S0["o1", "o2"][:, None]) ** 2) ** 2
)
plt.title(f"{rmse=}")
plt.show()
# -

# ### Compact MZI
#
# Let's see if we can improve variability (i.e. the RMSE w.r.t. nominal) by making the MZI more compact:
#


@gf.cell
def compact_mzi():
    c = gf.Component()

    # instances
    mmi_in = gf.components.mmi1x2()
    mmi_out = gf.components.mmi2x2()
    bend = gf.components.bend_euler()
    half_delay_straight = gf.components.straight()
    middle_straight = gf.components.straight(length=6.0)
    half_middle_straight = gf.components.straight(3.0)

    # references (sax convention: vars ending in underscore are references)
    mmi_in_ = c << mmi_in

    bend_top1_ = c << bend
    straight_top1_ = c << half_delay_straight
    bend_top2_ = (c << bend).mirror()
    straight_top2_ = c << middle_straight
    bend_top3_ = (c << bend).mirror()
    straight_top3_ = c << half_delay_straight
    bend_top4_ = c << bend

    straight_btm1_ = c << half_middle_straight
    bend_btm1_ = c << bend
    bend_btm2_ = (c << bend).mirror()
    bend_btm3_ = (c << bend).mirror()
    bend_btm4_ = c << bend
    straight_btm2_ = c << half_middle_straight

    mmi_out_ = c << mmi_out

    # connections
    bend_top1_.connect("o1", mmi_in_.ports["o2"])
    straight_top1_.connect("o1", bend_top1_.ports["o2"])
    bend_top2_.connect("o1", straight_top1_.ports["o2"])
    straight_top2_.connect("o1", bend_top2_.ports["o2"])
    bend_top3_.connect("o1", straight_top2_.ports["o2"])
    straight_top3_.connect("o1", bend_top3_.ports["o2"])
    bend_top4_.connect("o1", straight_top3_.ports["o2"])

    straight_btm1_.connect("o1", mmi_in_.ports["o3"])
    bend_btm1_.connect("o1", straight_btm1_.ports["o2"])
    bend_btm2_.connect("o1", bend_btm1_.ports["o2"])
    bend_btm3_.connect("o1", bend_btm2_.ports["o2"])
    bend_btm4_.connect("o1", bend_btm3_.ports["o2"])
    straight_btm2_.connect("o1", bend_btm4_.ports["o2"])

    mmi_out_.connect("o1", straight_btm2_.ports["o2"])

    # ports
    c.add_port(
        "o1",
        port=mmi_in_.ports["o1"],
    )
    c.add_port("o2", port=mmi_out_.ports["o3"])
    c.add_port("o3", port=mmi_out_.ports["o4"])
    return c


compact_mzi1 = compact_mzi()
fig = compact_mzi1.plot()
placements = compact_mzi1.get_netlist()["placements"]
mzi3, _ = sax.circuit(compact_mzi1.get_netlist(), models=models)

# +
mzi_params = sax.get_settings(mzi3)
placements = compact_mzi1.get_netlist()["placements"]
width_params = {
    k: {"width": widths(xm, ym, wmaps, v["x"], v["y"])}
    for k, v in placements.items()
    if "width" in mzi_params[k]
}

S0 = mzi3(wl=wl)
S = mzi3(
    wl=wl[:, None],
    **width_params,
)
ps = plt.plot(wl * 1e3, abs(S["o1", "o2"]) ** 2, color="C0", lw=1, alpha=0.1)
nps = plt.plot(wl * 1e3, abs(S0["o1", "o2"]) ** 2, color="C1", lw=2, alpha=1)
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.plot([1550, 1550], [-1, 2], color="black", ls=":")
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.figlegend([*ps[-1:], *nps], ["MC", "nominal"], bbox_to_anchor=(1.1, 0.9))
rmse = jnp.mean(
    jnp.abs(jnp.abs(S["o1", "o2"]) ** 2 - jnp.abs(S0["o1", "o2"][:, None]) ** 2) ** 2
)
plt.title(f"{rmse=}")
plt.show()
# -

# ## Phase shifter model
#
# You can create a phase shifter model that depends on the applied volage.
# For that you need first to figure out what's the phase shift for different voltages.

delta_length = 10
mzi_component = gf.components.mzi_phase_shifter_top_heater_metal(
    delta_length=delta_length
)
mzi_component


# +
def straight(wl=1.5, length=10.0, neff=2.4) -> sax.SDict:
    return sax.reciprocal({("o1", "o2"): jnp.exp(2j * jnp.pi * neff * length / wl)})


def mmi1x2() -> sax.SDict:
    """Returns a perfect 1x2 splitter."""
    return sax.reciprocal(
        {
            ("o1", "o2"): 0.5**0.5,
            ("o1", "o3"): 0.5**0.5,
        }
    )


def bend_euler(wl=1.5, length=20.0) -> sax.SDict:
    """Returns bend Sparameters with reduced transmission compared to a straight."""
    return {k: 0.99 * v for k, v in straight(wl=wl, length=length).items()}


def phase_shifter_heater(
    wl: float = 1.55,
    neff: float = 2.34,
    voltage: float = 0,
    length: float = 10,
    loss: float = 0.0,
) -> sax.SDict:
    """Returns simple phase shifter model"""
    deltaphi = voltage * jnp.pi
    phase = 2 * jnp.pi * neff * length / wl + deltaphi
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    return sax.reciprocal(
        {
            ("o1", "o2"): transmission,
        }
    )


models = {
    "bend_euler": bend_euler,
    "mmi1x2": mmi1x2,
    "straight": straight,
    "straight_heater_metal_undercut": phase_shifter_heater,
}
# -

mzi_component = gf.components.mzi_phase_shifter_top_heater_metal(
    delta_length=delta_length
)
netlist = mzi_component.get_netlist()
mzi_circuit, _ = sax.circuit(netlist=netlist, models=models)
S = mzi_circuit(wl=1.55)
S

# +
wl = np.linspace(1.5, 1.6, 256)
S = mzi_circuit(wl=wl)

plt.figure(figsize=(14, 4))
plt.title("MZI")
plt.plot(1e3 * wl, jnp.abs(S["o1", "o2"]) ** 2)
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.grid(True)
plt.show()
# -

# Now you can tune the phase shift applied to one of the arms.
#
# How do you find out what's the name of the netlist component that you want to tune?
#
# You can backannotate the netlist and read the labels on the backannotated netlist or you can plot the netlist

mzi_component.plot_netlist()

# As you can see the top phase shifter instance name `sxt` is hard to see on the netlist.
# You can also reconstruct the component using the netlist and look at the labels in klayout.

mzi_yaml = mzi_component.get_netlist_yaml()
mzi_component2 = gf.read.from_yaml(mzi_yaml)
mzi_component2.plot(label_aliases=True)

# The best way to get a deterministic name of the `instance` is naming the reference on your Pcell.

# +
voltages = np.linspace(-1, 1, num=5)
voltages = [-0.5, 0, 0.5]

for voltage in voltages:
    S = mzi_circuit(
        wl=wl,
        sxt={"voltage": voltage},
    )
    plt.plot(wl * 1e3, abs(S["o1", "o2"]) ** 2, label=str(voltage))
    plt.xlabel("λ [nm]")
    plt.ylabel("T")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)

plt.title("MZI vs voltage")
plt.legend()
# -

# ## Optimization
#
# You can optimize an MZI to get T=0 at 1530nm.
# To do this, you need to define a loss function for the circuit at 1550nm.
# This function should take the parameters that you want to optimize as positional arguments:
#


# +
def straight(wl=1.5, length=10.0, neff=2.4) -> sax.SDict:
    return sax.reciprocal({("o1", "o2"): jnp.exp(2j * jnp.pi * neff * length / wl)})


def mmi1x2():
    """Assumes a perfect 1x2 splitter"""
    return sax.reciprocal(
        {
            ("o1", "o2"): 0.5**0.5,
            ("o1", "o3"): 0.5**0.5,
        }
    )


def bend_euler(wl=1.5, length=20.0):
    """ "Let's assume a reduced transmission for the euler bend compared to a straight"""
    return {k: 0.99 * v for k, v in straight(wl=wl, length=length).items()}


models = {
    "bend_euler": bend_euler,
    "mmi1x2": mmi1x2,
    "straight": straight,
}
# -

delta_length = 30
mzi_component = gf.components.mzi(delta_length=delta_length)
mzi_circuit, _ = sax.circuit(netlist=mzi_component.get_netlist(), models=models)
S = mzi_circuit(wl=1.55)
S

# +
wl = np.linspace(1.5, 1.6, 256)
S = mzi_circuit(wl=wl)

plt.figure(figsize=(14, 4))
plt.title("MZI")
plt.plot(1e3 * wl, jnp.abs(S["o1", "o2"]) ** 2)
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.plot([1530, 1530], [0, 1])
plt.grid(True)
plt.show()
# -

# GDSFactory autonames component names for GDS and for netlists uses an incremental name for easier addressing of the references.

netlist = mzi_component.get_netlist()
c = gf.read.from_yaml(netlist)
c.plot()

# From this we see that we will need to change `syl` and `straight_9`.

# +
mzi_component = gf.components.mzi(
    delta_length=delta_length,
)
mzi_circuit, _ = sax.circuit(
    netlist=mzi_component.get_netlist(),
    models=models,
)


@jax.jit
def loss_fn(delta_length):
    S = mzi_circuit(
        wl=1.53,
        syl={
            "length": delta_length / 2 + 2,
        },
        straight_9={
            "length": delta_length / 2 + 2,
        },
    )
    return (abs(S["o1", "o2"]) ** 2).mean()


# -

# %time loss_fn(20.0)

# You can use this loss function to define a grad function which works on the parameters of the loss function:

grad_fn = jax.jit(
    jax.grad(
        loss_fn,
        argnums=0,  # JAX gradient function for the first positional argument, jitted
    )
)

# Next, you need to define a JAX optimizer, which on its own is nothing more than three more functions:
#
# 1. an initialization function with which to initialize the optimizer state
# 2. an update function which will update the optimizer state (and with it the model parameters).
# 3. a function with the model parameters given the optimizer state.

initial_delta_length = 30.0
init_fn, update_fn, params_fn = opt.adam(step_size=0.1)
state = init_fn(initial_delta_length)


def step_fn(step, state):
    settings = params_fn(state)
    loss = loss_fn(settings)
    grad = grad_fn(settings)
    state = update_fn(step, grad, state)
    return loss, state


range_ = trange(100)
for step in range_:
    loss, state = step_fn(step, state)
    range_.set_postfix(loss=f"{loss:.6f}")

delta_length = params_fn(state)
delta_length

S = mzi_circuit(
    wl=wl,
    syl={"length": delta_length / 2 + 2},
    straight_9={"length": delta_length / 2 + 2},
)
plt.figure(figsize=(14, 4))
plt.plot(wl * 1e3, abs(S["o1", "o2"]) ** 2)
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.ylim(-0.05, 1.05)
plt.plot([1530, 1530], [0, 1])
plt.grid(True)
plt.show()


# The minimum of the MZI is perfectly located at 1530nm.

# ## Hierarchical circuits
#
# You can also simulate hierarchical circuits, such as lattice of MZI interferometers.
#


# +
@gf.cell
def mzis(delta_length=10):
    c = gf.Component()
    c1 = c << gf.components.mzi(delta_length=delta_length)
    c2 = c << gf.components.mzi(delta_length=delta_length)
    c2.connect("o1", c1.ports["o2"])

    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c2.ports["o2"])
    return c


def straight(wl=1.5, length=10.0, neff=2.4) -> sax.SDict:
    """Straight model."""
    return sax.reciprocal({("o1", "o2"): jnp.exp(2j * jnp.pi * neff * length / wl)})


def mmi1x2():
    """Assumes a perfect 1x2 splitter."""
    return sax.reciprocal(
        {
            ("o1", "o2"): 0.5**0.5,
            ("o1", "o3"): 0.5**0.5,
        }
    )


def bend_euler(wl=1.5, length=20.0):
    """Assumes reduced transmission for the euler bend compared to a straight."""
    return {k: 0.99 * v for k, v in straight(wl=wl, length=length).items()}


models = {
    "bend_euler": bend_euler,
    "mmi1x2": mmi1x2,
    "straight": straight,
}


c2 = mzis()
c2
# -

c2.plot_netlist_flat()

c1 = gf.components.mzi(delta_length=10)
c1

c1.plot_netlist()

# +
wl = np.linspace(1.5, 1.6)
netlist1 = c1.get_netlist_recursive()
circuit1, _ = sax.circuit(netlist=netlist1, models=models)
S1 = circuit1(wl=wl)

netlist2 = c2.get_netlist_recursive()
circuit2, _ = sax.circuit(netlist=netlist2, models=models)
S2 = circuit2(wl=wl)

plt.figure(figsize=(14, 4))
plt.plot(1e3 * wl, jnp.abs(S1["o1", "o2"]) ** 2, label="1 MZI")
plt.plot(1e3 * wl, jnp.abs(S2["o1", "o2"]) ** 2, label="2 MZI")
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.grid(True)
plt.legend()
plt.show()
