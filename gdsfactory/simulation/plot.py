from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import gdsfactory as gf


def plot_sparameters(
    sp: Dict[str, np.ndarray],
    logscale: bool = True,
    keys: Optional[Tuple[str, ...]] = None,
    with_simpler_input_keys: bool = False,
    with_simpler_labels: bool = True,
) -> None:
    """Plots Sparameters from a dict of np.ndarrays.

    Args:
        sp: Sparameters np.ndarray.
        logscale: plots 20*log10(S).
        keys: list of keys to plot, plots all by default.
        with_simpler_input_keys: You can use S12 keys instead of o1@0,o2@0.
        with_simpler_labels: uses S11, S12 in plot labels instead of o1@0,o2@0.

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
        if with_simpler_input_keys:
            key = f"o{key[1]}@0,o{key[2]}@0"
            if key not in sp:
                raise ValueError(f"{key!r} not in {list(sp.keys())}")

        if with_simpler_labels and "o" in key and "@" in key:
            port_mode1_port_mode2 = key.split(",")
            if len(port_mode1_port_mode2) != 2:
                raise ValueError(f"{key!r} needs to be 'portin@mode,portout@mode'")
            port_mode1, port_mode2 = port_mode1_port_mode2
            port1, _mode1 = port_mode1.split("@")
            port2, _mode2 = port_mode2.split("@")
            alias = f"S{port1[1:]}{port2[1:]}"
        else:
            alias = key

        if key not in sp:
            raise ValueError(f"{key!r} not in {list(sp.keys())}")
        y = sp[key]
        y = 20 * np.log10(np.abs(y)) if logscale else np.abs(y) ** 2
        plt.plot(w, y, label=alias)
    plt.legend()
    plt.xlabel("wavelength (nm)")
    plt.ylabel("|S| (dB)") if logscale else plt.ylabel("$|S|^2$")
    plt.show()


def plot_sparameters_phase(
    sp: Dict[str, np.ndarray],
    logscale: bool = True,
    keys: Optional[Tuple[str, ...]] = None,
    with_simpler_input_keys: bool = False,
    with_simpler_labels: bool = True,
) -> None:
    w = sp["wavelengths"] * 1e3
    keys = keys or [key for key in sp if not key.lower().startswith("wav")]

    for key in keys:
        if with_simpler_input_keys:
            key = f"o{key[1]}@0,o{key[2]}@0"
            if key not in sp:
                raise ValueError(f"{key!r} not in {list(sp.keys())}")

        if with_simpler_labels and "o" in key and "@" in key:
            port_mode1_port_mode2 = key.split(",")
            if len(port_mode1_port_mode2) != 2:
                raise ValueError(f"{key!r} needs to be 'portin@mode,portout@mode'")
            port_mode1, port_mode2 = port_mode1_port_mode2
            port1, _mode1 = port_mode1.split("@")
            port2, _mode2 = port_mode2.split("@")
            alias = f"S{port1[1:]}{port2[1:]}"
        else:
            alias = key

        if key not in sp:
            raise ValueError(f"{key!r} not in {list(sp.keys())}")
        y = sp[key]
        y = np.angle(y)
        plt.plot(w, y, label=alias)
    plt.legend()
    plt.xlabel("wavelength (nm)")
    plt.ylabel("S (deg)")
    plt.show()


def plot_imbalance2x2(
    sp: Dict[str, np.ndarray], port1: str = "o1@0,o3@0", port2: str = "o1@0,o4@0"
) -> None:
    """Plots imbalance in % for 2x2 coupler.

    Args:
        sp: sparameters dict np.ndarray.
        port1: port1Name@modeIndex.
        port2: port2Name@modeIndex.

    """
    if port1 not in sp:
        raise ValueError(f"{port1!r} not in {list(sp.keys())}")

    if port2 not in sp:
        raise ValueError(f"{port2!r} not in {list(sp.keys())}")

    y1 = np.abs(sp[port1])
    y2 = np.abs(sp[port2])
    imbalance = y1 / y2
    x = sp["wavelengths"] * 1e3
    plt.plot(x, abs(imbalance))
    plt.xlabel("wavelength (nm)")
    plt.ylabel("imbalance (%)")
    plt.grid()


def plot_loss2x2(
    sp: Dict[str, np.ndarray], port1: str = "o1@0,o3@0", port2: str = "o1@0,o4@0"
) -> None:
    """Plots imbalance in % for 2x2 coupler.

    Args:
        sp: sparameters dict np.ndarray.
        port1: port name @ mode index. o1@0 is the fundamental mode for o1 port.
        port2: port name @ mode index. o1@0 is the fundamental mode for o1 port.

    """
    if port1 not in sp:
        raise ValueError(f"{port1!r} not in {list(sp.keys())}")

    if port2 not in sp:
        raise ValueError(f"{port2!r} not in {list(sp.keys())}")
    y1 = np.abs(sp[port1])
    y2 = np.abs(sp[port2])
    x = sp["wavelengths"] * 1e3
    plt.plot(x, abs(10 * np.log10(y1**2 + y2**2)))
    plt.xlabel("wavelength (nm)")
    plt.ylabel("excess loss (dB)")


plot_loss1x2 = partial(plot_loss2x2, port1="o1@0,o2@0", port2="o1@0,o3@0")
plot_imbalance1x2 = partial(plot_imbalance2x2, port1="o1@0,o2@0", port2="o1@0,o3@0")


if __name__ == "__main__":
    import gdsfactory.simulation as sim

    sp = sim.get_sparameters_data_tidy3d(component=gf.components.mmi1x2)
    # plot_sparameters(sp, logscale=False, keys=["o1@0,o2@0"])
    # plot_sparameters(sp, logscale=False, keys=["S21"])
    # plt.show()
