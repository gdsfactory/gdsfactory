import numpy as np
import pandas as pd
from sax.typing_ import Float, SDict
from scipy.interpolate import interp1d

from gdsfactory.simulation.get_sparameters_path import get_sparameters_path_lumerical
from gdsfactory.types import PathType

wl_cband = np.linspace(1.500, 1.600, 128)


def from_csv(
    filepath: PathType,
    wl: Float = wl_cband,
    xkey: str = "wavelength",
    xunits: float = 1.0,
    prefix: str = "S",
) -> SDict:
    """loads Sparameters from a CSV file
    Returns interpolated Sdict over wl

    Args:
        filepath:
        wl: wavelength to interpolate
        xkey: key for wavelengths in file
        xunits: x units in um from the loaded file
        prefix: for the sparameters column names in file
    """
    df = pd.read_csv(filepath)
    nsparameters = (len(df.keys()) - 1) // 2
    nports = int(nsparameters ** 0.5)

    x = df[xkey] * xunits

    s = {}
    for i in range(1, nports + 1):
        for j in range(1, nports + 1):
            s[f"{prefix}{i}{j}m"] = interp1d(x, df[f"{prefix}{i}{j}m"])(wl)
            s[f"{prefix}{i}{j}a"] = interp1d(x, df[f"{prefix}{i}{j}a"])(wl)

    return {
        (f"o{i}", f"o{j}"): s[f"{prefix}{i}{j}m"] * np.exp(1j * s[f"{prefix}{i}{j}a"])
        for i in range(1, nports + 1)
        for j in range(1, nports + 1)
    }


def demo_mmi_lumerical_csv():
    import matplotlib.pyplot as plt

    filepath = get_sparameters_path_lumerical(gf.c.mmi1x2)
    mmi = partial(from_csv, filepath=filepath, xunits=1e-3, xkey="wavelength")
    sax.plot.plot_model(mmi)
    plt.show()


if __name__ == "__main__":
    from functools import partial

    import matplotlib.pyplot as plt
    import sax

    import gdsfactory as gf

    # s = from_csv(filepath=filepath, wl=wl_cband, xunits=1e-3)
    # s21 = s[("o1", "o3")]
    # plt.plot(wl_cband, np.abs(s21) ** 2)
    # plt.show()

    filepath = get_sparameters_path_lumerical(gf.c.mmi1x2)
    mmi = partial(from_csv, filepath=filepath, xunits=1e-3, xkey="wavelength")
    sax.plot.plot_model(mmi)
    plt.show()

    # model = partial(
    #     from_csv, filepath=filepath, xunits=1, xkey="wavelengths", prefix="s"
    # )
    # sax.plot.plot_model(model)
    # plt.show()
