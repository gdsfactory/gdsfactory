"""Compute and write Sparameters using Meep
"""

import pathlib
from pathlib import Path, PosixPath
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import pandas as pd
import pp
from pp.component import Component
from pp.sp.get_sparameters_path import get_sparameters_path

from gmeep.config import PATH
from gmeep.get_simulation import (
    get_simulation,
    LAYER_TO_THICKNESS_NM,
    LAYER_TO_MATERIAL,
)


def write_sparameters(
    component: Component,
    dirpath: PosixPath = PATH.sparameters,
    layer_to_thickness_nm: Dict[Tuple[int, int], float] = LAYER_TO_THICKNESS_NM,
    layer_to_material: Dict[Tuple[int, int], str] = LAYER_TO_MATERIAL,
    filepath: Optional[Path] = None,
    overwrite: bool = False,
    **settings,
) -> pd.DataFrame:
    """Compute Sparameters and writes them in CSV filepath.

    Args:
        component: to simulate.
        dirpath: directory to store Sparameters
        layer_to_thickness_nm: GDS layer (int, int) to thickness
        layer_to_material: GDS layer (int, int) to material string ('Si', 'SiO2', ...)
        filepath: to store pandas Dataframe with Sparameters in CSV format
        overwrite: overwrites
        **settings: sim settings

    Returns:
        sparameters in a pandas Dataframe

    """
    filepath = filepath or get_sparameters_path(
        component=component,
        dirpath=dirpath,
        layer_to_material=layer_to_material,
        layer_to_thickness_nm=layer_to_thickness_nm,
    )
    filepath = pathlib.Path(filepath)
    if filepath.exists() and not overwrite:
        print(f"Simulation loaded from file: {filepath}")
        return pd.read_csv(filepath)

    sim_dict = get_simulation(
        component=component,
        layer_to_thickness_nm=layer_to_thickness_nm,
        layer_to_material=layer_to_material,
        **settings,
    )

    sim = sim_dict["sim"]
    monitors = sim_dict["monitors"]
    freqs = sim_dict["freqs"]
    field_monitor_point = sim_dict["field_monitor_point"]
    wavelengths = 1 / freqs

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            dt=50, c=mp.Ez, pt=field_monitor_point, decay_by=1e-9
        )
    )
    # call this function every 50 time spes
    # look at simulation and measure component that we want to measure (Ez component)
    # when field_monitor_point decays below a certain 1e-9 field threshold

    # Calculate mode overlaps
    nports = len(monitors)
    S = np.zeros((len(freqs), nports, nports))
    a = {}
    b = {}

    for port_name, monitor in monitors.items():
        m_results = sim.get_eigenmode_coefficients(monitor, [1]).alpha

        # Parse out the overlaps
        a[port_name] = m_results[:, :, 0]  # forward wave
        b[port_name] = m_results[:, :, 1]  # backward wave

    for i, port_name_i in enumerate(monitors.keys()):
        for j, port_name_j in enumerate(monitors.keys()):
            S[:, i, j] = np.squeeze(a[port_name_j] / b[port_name_i])
            S[:, j, i] = np.squeeze(a[port_name_i] / b[port_name_j])

    # for port_name in monitor.keys():
    #     a1 = m1_results[:, :, 0]  # forward wave
    #     b1 = m1_results[:, :, 1]  # backward wave
    #     a2 = m2_results[:, :, 0]  # forward wave
    #     # b2 = m2_results[:, :, 1]  # backward wave

    #     # Calculate the actual scattering parameters from the overlaps
    #     s11 = np.squeeze(b1 / a1)
    #     s12 = np.squeeze(a2 / a1)

    r = dict(wavelengths=wavelengths)
    keys = [key for key in r.keys() if key.startswith("s")]
    s = {f"{key}a": list(np.unwrap(np.angle(r[key].flatten()))) for key in keys}
    s.update({f"{key}m": list(np.abs(r[key].flatten())) for key in keys})
    s.update(wavelengths=wavelengths)
    s.update(freqs=freqs)
    df = pd.DataFrame(s)
    # df = df.set_index(df.wavelength)
    df.to_csv(filepath, index=False)

    return df


def plot_sparameters(df: pd.DataFrame) -> None:
    """Plot Sparameters from a Pandas DataFrame."""
    wavelengths = df["wavelengths"]
    for key in df.keys():
        if key.endswith("m"):
            plt.plot(
                wavelengths,
                df[key],
                "-o",
                label="key",
            )
    plt.ylabel("Power (dB)")
    plt.xlabel(r"Wavelength ($\mu$m)")
    plt.legend()
    plt.grid(True)


def write_sparameters_sweep(
    component,
    dirpath: PosixPath = PATH.sparameters,
    layer_to_thickness_nm: Dict[Tuple[int, int], float] = {(1, 0): 220.0},
    layer_to_material: Dict[Tuple[int, int], str] = LAYER_TO_MATERIAL,
    **kwargs,
):
    """From gdsfactory component writes Sparameters for all the ports
    Returns the full Sparameters matrix
    """
    filepath_lumerical = get_sparameters_path(
        component=component,
        dirpath=dirpath,
        layer_to_material=layer_to_material,
        layer_to_thickness_nm=layer_to_thickness_nm,
    )
    for port_source_name in component.ports.keys():
        sim_dict = get_simulation(
            port_source_name=port_source_name,
            layer_to_thickness_nm=layer_to_thickness_nm,
            layer_to_material=layer_to_material,
            **kwargs,
        )
        filepath = filepath_lumerical.with_suffix(f"{port_source_name}.csv")
        write_sparameters(sim_dict, filepath=filepath)


if __name__ == "__main__":

    c = pp.c.bend_circular(radius=2)
    c = pp.add_padding(c, default=0, bottom=2, right=2, layers=[(100, 0)])

    c = pp.c.mmi1x2()
    c = pp.add_padding(c, default=0, bottom=2, top=2, layers=[(100, 0)])

    c = pp.c.straight(length=2)
    c = pp.add_padding(c, default=0, bottom=2, top=2, layers=[(100, 0)])

    sim_dict = get_simulation(c, is_3d=False)
    df = write_sparameters(c, overwrite=True)
    plot_sparameters(df)
    plt.show()
