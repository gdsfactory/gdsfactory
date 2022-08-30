from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import gdsfactory as gf


def plot_sparameters(
    sp: Dict[str, np.ndarray],
    logscale: bool = True,
    keys: Optional[Tuple[str, ...]] = None,
    with_sparameter_labels: bool = True,
    with_sparameter_keys: bool = False,
) -> None:
    """Plots Sparameters from a dict of np.ndarrays.

    Args:
        sp: Sparameters np.ndarray.
        logscale: plots 20*log10(S).
        keys: list of keys to plot, plots all by default.
        with_sparameter_keys: uses S11, S12 in plot labels.
        with_sparameter_keys: use Sparameter keys.
            Assumes mode 0 and o1, o2, o3 port naming.

    .. plot::
        :include-source:

        import gdsfactory as gf
        import gdsfactory.simulation as sim

        sp = sim.get_sparameters_data_lumerical(component=gf.components.mmi1x2)
        sim.plot.plot_sparameters(sp, logscale=True)

    """
    w = sp["wavelengths"] * 1e3
    keys = keys or [key for key in sp if not key.lower().startswith("wav")]

    for key in keys:

        if with_sparameter_keys:
            key = f"o{key[1]}@0,o{key[2]}@0"

        if with_sparameter_labels and "o" in key and "@" in key:
            port_mode1, port_mode2 = key.split(",")
            port1, _mode1 = port_mode1.split("@")
            port2, _mode2 = port_mode2.split("@")
            alias = f"S{port1[1:]}{port2[1:]}"
        else:
            alias = key

        if key not in sp:
            raise ValueError(f"{key} not in {sp.keys()}")
        y = sp[key]
        y = 20 * np.log10(np.abs(y)) if logscale else np.abs(y) ** 2
        plt.plot(w, y, label=alias)
    plt.legend()
    plt.xlabel("wavelength (nm)")
    plt.ylabel("|S| (dB)") if logscale else plt.ylabel("$|S|^2$")
    plt.show()


def plot_imbalance2x2(
    sp: Dict[str, np.ndarray], port1: str = "s13m", port2: str = "s14m"
) -> None:
    """Plots imbalance in % for 2x2 coupler.

    Args:
        sp: sparameters dict np.ndarray.
        port1: name.
        port2: name.

    """
    y1 = sp[port1].values
    y2 = sp[port2].values
    imbalance = y1 / y2
    x = sp["wavelengths"] * 1e3
    plt.plot(x, 100 * abs(imbalance))
    plt.xlabel("wavelength (nm)")
    plt.ylabel("imbalance (%)")
    plt.grid()


def plot_loss2x2(
    sp: Dict[str, np.ndarray], port1: str = "s13m", port2: str = "s14m"
) -> None:
    """Plots imbalance in % for 2x2 coupler.

    Args:
        sp: sparameters dict np.ndarray.
        port1: name.
        port2: name.

    """
    y1 = sp[port1].values
    y2 = sp[port2].values
    x = sp["wavelengths"] * 1e3
    plt.plot(x, abs(10 * np.log10(y1**2 + y2**2)))
    plt.xlabel("wavelength (nm)")
    plt.ylabel("excess loss (dB)")


plot_loss1x2 = gf.partial(plot_loss2x2, port1="s13m", port2="o1@0,o2@0")
plot_imbalance1x2 = gf.partial(plot_imbalance2x2, port1="s13m", port2="o1@0,o2@0")


if __name__ == "__main__":
    import gdsfactory.simulation as sim

    sp = sim.get_sparameters_data_tidy3d(component=gf.components.mmi1x2)
    plot_sparameters(sp, logscale=False, keys=["o1@0,o2@0"])
    plt.show()
