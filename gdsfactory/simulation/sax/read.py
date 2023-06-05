"""read Sparameters from CSV file and returns sax model."""
from __future__ import annotations

import pathlib
from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sax.typing_ import Float, Model
from typing_extensions import Literal

import gdsfactory as gf
from gdsfactory.simulation.get_sparameters_path import (
    get_sparameters_path_lumerical,
    get_sparameters_path_meep,
    get_sparameters_path_tidy3d,
)

wl_cband = np.linspace(1.500, 1.600, 128)

PathType = Union[str, pathlib.Path]

Simulator = Literal["lumerical", "meep", "tidy3d"]


def model_from_npz(
    filepath: Union[PathType, np.ndarray],
    xkey: str = "wavelengths",
    xunits: float = 1,
    prefix: str = "s",
) -> Model:
    """Returns a SAX Sparameters Model from a npz file.

    The SAX Model is a function that returns a SAX SDict interpolated over wavelength.

    Args:
        filepath: CSV Sparameters path or pandas DataFrame.
        wl: wavelength to interpolate (um).
        xkey: key for wavelengths in file.
        xunits: x units in um from the loaded file (um). 1 means 1um.
        prefix: for the sparameters column names in file.

    """
    sp = np.load(filepath) if isinstance(filepath, (pathlib.Path, str)) else filepath
    keys = list(sp.keys())

    if xkey not in keys:
        raise ValueError(f"{xkey!r} not in {keys}")

    x = jnp.asarray(sp[xkey] * xunits)
    wl = jnp.asarray(wl_cband)

    # make sure x is sorted from low to high
    idxs = jnp.argsort(x)
    x = x[idxs]
    sp = {k: v[idxs] for k, v in sp.items()}

    @jax.jit
    def model(wl: Float = wl):
        S = {}
        zero = jnp.zeros_like(x)

        for key in sp:
            if not key.startswith("wav"):
                port_mode0, port_mode1 = key.split(",")
                port0, mode0 = port_mode0.split("@")
                port1, mode1 = port_mode1.split("@")
                S[(port0, port1)] = jnp.interp(wl, x, sp.get(key, zero))

        return S

    return model


def model_from_csv(
    filepath: Union[PathType, pd.DataFrame],
    xkey: str = "wavelengths",
    xunits: float = 1,
    prefix: str = "s",
) -> Model:
    """Returns a SAX Sparameters Model from a CSV file.

    The SAX Model is a function that returns a SAX SDict interpolated over wavelength.

    Args:
        filepath: CSV Sparameters path or pandas DataFrame.
        wl: wavelength to interpolate (um).
        xkey: key for wavelengths in file.
        xunits: x units in um from the loaded file (um). 1 means 1um.
        prefix: for the sparameters column names in file.
    """
    df = filepath if isinstance(filepath, pd.DataFrame) else pd.read_csv(filepath)
    assert isinstance(df, pd.DataFrame)
    df = df.reset_index()  # maybe there is useful info in the index...
    dic = dict(zip(df.columns, jnp.asarray(df.values.T)))
    keys = list(dic.keys())

    if xkey not in keys:
        raise ValueError(f"{xkey!r} not in {keys}")

    nsparameters = (len(keys) - 1) // 2
    nports = int(nsparameters**0.5)

    x = jnp.asarray(dic[xkey] * xunits)
    wl = jnp.asarray(wl_cband)

    # make sure x is sorted from low to high
    idxs = jnp.argsort(x)
    x = x[idxs]
    dic = {k: v[idxs] for k, v in dic.items()}

    @jax.jit
    def model(wl: Float = wl):
        S = {}
        zero = jnp.zeros_like(x)
        for i in range(1, nports + 1):
            for j in range(1, nports + 1):
                m = jnp.interp(wl, x, dic.get(f"{prefix}{i}{j}m", zero))
                a = jnp.interp(wl, x, dic.get(f"{prefix}{i}{j}a", zero))
                S[f"o{i}", f"o{j}"] = m * jnp.exp(1j * a)

        return S

    return model


def _demo_mmi_lumerical_csv() -> None:
    import matplotlib.pyplot as plt
    from plot_model import plot_model

    filepath = get_sparameters_path_lumerical(gf.c.mmi1x2)
    mmi = model_from_csv(filepath=filepath, xkey="wavelengths")
    plot_model(mmi)
    plt.show()


def model_from_component(component, simulator: Simulator, **kwargs) -> Model:
    """Returns SAX model from lumerical FDTD simulations.

    Args:
        component: to simulate.
        simulator: meep, lumerical or tidy3d.
        kwargs: simulator settings.

    """
    simulators = ["lumerical", "meep", "tidy3d"]

    if simulator == "lumerical":
        filepath = get_sparameters_path_lumerical(component=component, **kwargs)
    elif simulator == "meep":
        filepath = get_sparameters_path_meep(component=component, **kwargs)
    elif simulator == "tidy3d":
        filepath = get_sparameters_path_tidy3d(component=component, **kwargs)
    else:
        raise ValueError(f"{simulator!r} no in {simulators}")
    return model_from_csv(filepath=filepath)


model_from_component_lumerical = partial(model_from_component, simulator="fdtd")


if __name__ == "__main__":
    # mmi1x2 = partial(model_from_component_lumerical, component=gf.components.mmi1x2)
    # mmi2x2 = partial(model_from_component_lumerical, component=gf.components.mmi2x2)

    # grating_coupler_elliptical = model_from_csv(
    #     filepath=sparameters_path / "grating_coupler_ellipti_9d85a0c6_18c08cac.csv"
    # )

    # model_factory = dict(
    #     mmi1x2=mmi1x2, mmi2x2=mmi2x2, grating_coupler_elliptical=grating_coupler_elliptical
    # )
    # from gdsfactory.config import sparameters_path

    import matplotlib.pyplot as plt
    from plot_model import plot_model

    # s = from_csv(filepath=filepath, wavelength=wl_cband, xunits=1e-3)
    # s21 = s[("o1", "o3")]
    # plt.plot(wl_cband, np.abs(s21) ** 2)
    # plt.show()
    # filepath = get_sparameters_path_lumerical(gf.components.mmi1x2)
    # mmi = model_from_csv(filepath=filepath, xkey="wavelengths")
    # plot_model(mmi)
    # plt.show()
    # plot_model(grating_coupler_elliptical)
    # plt.show()
    # model = partial(
    #     from_csv, filepath=filepath, xunits=1, xkey="wavelengths", prefix="s"
    # )
    # sax.plot.plot_model(model)
    # plt.show()
    #  This looks correct
    # coupler_fdtd = partial(
    #     sdict_from_csv,
    #     filepath=gf.config.sparameters_path / "coupler" / "coupler_G224n_L20_S220.csv",
    #     xkey="wavelength_nm",
    #     prefix="S",
    #     xunits=1e-3,
    # )
    # this looks wrong
    # coupler_fdtd = model_from_csv(
    #     filepath=sparameters_path / "coupler" / "coupler_G224n_L20_S220.csv",
    #     xkey="wavelength_nm",
    #     prefix="S",
    #     xunits=1e-3,
    # )
    # coupler_fdtd = model_from_npz(
    #     filepath=sparameters_path / "coupler" / "coupler_G224n_L20_S220.npz",
    #     xkey="wavelength_nm",
    #     prefix="S",
    #     xunits=1e-3,
    # )
    filepath = get_sparameters_path_tidy3d(gf.c.mmi1x2)
    coupler_fdtd = model_from_npz(filepath)
    plot_model(coupler_fdtd)
    plt.show()
