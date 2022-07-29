"""read Sparameters from a CSV file and returns sax model.

TODO: write dat to csv converter
(from lumerical interconnect to SAX format and the other way around)
"""

from functools import partial
from pathlib import PosixPath
from typing import Callable, Union

import numpy as np
import pandas as pd
from sax.typing_ import Float, SDict
from scipy.interpolate import interp1d

import gdsfactory as gf
from gdsfactory.config import sparameters_path
from gdsfactory.simulation.get_sparameters_path import get_sparameters_path_lumerical

wl_cband = np.linspace(1.500, 1.600, 128)

SDictFactory = Callable[..., SDict]


def sdict_from_csv(
    filepath: Union[str, PosixPath, pd.DataFrame],
    wl: Float = wl_cband,
    xkey: str = "wavelengths",
    xunits: float = 1,
    prefix: str = "s",
) -> SDict:
    """Returns SDict from Sparameters from a CSV file.

    Interpolates Sdict over wavelength.

    Args:
        filepath: CSV Sparameters path or pandas DataFrame.
        wl: wavelength to interpolate (um).
        xkey: key for wavelengths in file.
        xunits: x units in um from the loaded file (um). 1 means 1um.
        prefix: for the sparameters column names in file.
    """
    df = filepath if isinstance(filepath, pd.DataFrame) else pd.read_csv(filepath)
    keys = list(df.keys())

    if xkey not in df:
        raise ValueError(f"{xkey!r} not in {keys}")

    nsparameters = (len(keys) - 1) // 2
    nports = int(nsparameters**0.5)

    x = df[xkey] * xunits

    s = {}
    for i in range(1, nports + 1):
        for j in range(1, nports + 1):
            m = f"{prefix}{i}{j}m"
            a = f"{prefix}{i}{j}a"
            if m not in df:
                raise ValueError(f"{m!r} not in {keys}")
            if a not in df:
                raise ValueError(f"{a!r} not in {keys}")
            s[m] = interp1d(x, df[m])(wl)
            s[a] = interp1d(x, df[a])(wl)

    return {
        (f"o{i}", f"o{j}"): s[f"{prefix}{i}{j}m"] * np.exp(1j * s[f"{prefix}{i}{j}a"])
        for i in range(1, nports + 1)
        for j in range(1, nports + 1)
    }


def _demo_mmi_lumerical_csv() -> None:
    import matplotlib.pyplot as plt
    from plot_model import plot_model

    filepath = get_sparameters_path_lumerical(gf.c.mmi1x2)
    mmi = partial(sdict_from_csv, filepath=filepath, xkey="wavelengths")
    plot_model(mmi)
    plt.show()


def sdict_from_component_lumerical(component, **kwargs) -> SDictFactory:
    """Returns SDict based on lumerical simulation."""
    filepath = get_sparameters_path_lumerical(component=component, **kwargs)
    return partial(sdict_from_csv, filepath=filepath)


mmi1x2 = gf.partial(sdict_from_component_lumerical, component=gf.components.mmi1x2)
mmi2x2 = gf.partial(sdict_from_component_lumerical, component=gf.components.mmi2x2)

grating_coupler_elliptical = gf.partial(
    sdict_from_csv,
    filepath=sparameters_path / "grating_coupler_ellipti_9d85a0c6_18c08cac.csv",
)

model_factory = dict(
    mmi1x2=mmi1x2, mmi2x2=mmi2x2, grating_coupler_elliptical=grating_coupler_elliptical
)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from plot_model import plot_model

    # s = from_csv(filepath=filepath, wavelength=wl_cband, xunits=1e-3)
    # s21 = s[("o1", "o3")]
    # plt.plot(wl_cband, np.abs(s21) ** 2)
    # plt.show()
    # filepath = get_sparameters_path_lumerical(gf.c.mmi1x2)
    # mmi = partial(sdict_from_csv, filepath=filepath, xkey="wavelengths")
    # plot_model(mmi)
    # plt.show()

    plot_model(grating_coupler_elliptical)
    plt.show()

    # model = partial(
    #     from_csv, filepath=filepath, xunits=1, xkey="wavelengths", prefix="s"
    # )
    # sax.plot.plot_model(model)
    # plt.show()
