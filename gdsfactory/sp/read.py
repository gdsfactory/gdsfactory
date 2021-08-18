import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.sp.get_sparameters_path import get_sparameters_path
from gdsfactory.tech import LAYER_STACK, LayerStack


def get_ports(line: str) -> Tuple[str, str]:
    """Returns 2 port labels strings from interconnect file."""
    line = line.replace('"', "")
    line = line.replace("(", "")
    line_fields = line.split(",")
    port1 = line_fields[0]
    port2 = line_fields[3]
    return port1, port2


def read_sparameters_lumerical(
    filepath, numports: int
) -> Tuple[Tuple[str, ...], np.array, np.ndarray]:
    r"""Returns Sparameters from Lumerical interconnect export file.

    Args:
        filepath: Sparameters filepath (interconnect format)
        numports: number of ports

    Returns:
        port_names: list of port labels
        F: frequency 1d np.array
        S: Sparameters np.ndarray matrix

    """
    F = []
    S = []
    port_names = []

    with open(filepath, "r") as fid:
        for _i in range(numports):
            port_line = fid.readline()
            m = re.search(r'\[".*",', port_line)
            if m:
                port = m.group(0)
                port_names.append(port[2:-2])
        line = fid.readline()
        port1, port2 = get_ports(line)
        line = fid.readline()
        numrows = int(tuple(line[1:-2].split(","))[0])
        S = np.zeros((numrows, numports, numports), dtype="complex128")
        r = m = n = 0
        for line in fid:
            if line[0] == "(":
                if "transmission" in line:
                    port1, port2 = get_ports(line)
                continue
            data = line.split()
            data = list(map(float, data))
            if m == 0 and n == 0:
                F.append(data[0])

            i = port_names.index(port1)
            j = port_names.index(port2)
            S[r, i, j] = data[1] * np.exp(1j * data[2])
            r += 1
            if r == numrows:
                r = 0
                m += 1
                if m == numports:
                    m = 0
                    n += 1
                    if n == numports:
                        break

    # port_names.reverse()
    # print(len(F), S.shape, len(port_names))
    return (tuple(port_names), np.array(F), S)


def test_read_sparameters_2port_bend():
    filepath = gf.CONFIG["sp"] / "bend_circular" / "bend_circular_S220.dat"
    port_names, f, s = read_sparameters_lumerical(filepath=filepath, numports=2)
    print(port_names)
    assert port_names == ("N0", "W0")


def test_read_sparameters_2port_straight():
    filepath = gf.CONFIG["sp"] / "straight" / "straight_S220.dat"
    port_names, f, s = read_sparameters_lumerical(filepath=filepath, numports=2)
    print(port_names)
    assert len(f) == 500
    assert port_names == ("E0", "W0")


def test_read_sparameters_3port_mmi1x2():
    filepath = gf.CONFIG["sp"] / "mmi1x2" / "mmi1x2_S220.dat"
    port_names, f, s = read_sparameters_lumerical(filepath=filepath, numports=3)
    print(port_names)
    assert len(f) == 500
    assert port_names == ("E0", "E1", "W0")


def test_read_sparameters_4port_mmi2x2():
    filepath = gf.CONFIG["sp"] / "mmi2x2" / "mmi2x2_S220.dat"
    port_names, f, s = read_sparameters_lumerical(filepath=filepath, numports=4)
    print(port_names)
    assert len(f) == 500
    assert port_names == ("E0", "E1", "W0", "W1")


def read_sparameters_component(
    component: Component,
    layer_stack: LayerStack = LAYER_STACK,
    layer_to_material: Optional[Dict[Tuple[int, int], str]] = None,
    layer_to_thickness_nm: Optional[Dict[Tuple[int, int], int]] = None,
    dirpath: Path = gf.CONFIG["sp"],
) -> Tuple[List[str], np.array, np.ndarray]:
    r"""Returns Sparameters from Lumerical interconnect export file.

    Args:
        component: Component
        dirpath: path where to look for the Sparameters
        layer_to_material: layer to material dict
        layer_to_thickness_nm: layer to thickness (nm)

    Returns:
        port_names: list of port labels
        F: frequency 1d np.array
        S: Sparameters np.ndarray matrix


    the Sparameters file have Lumerical format
    https://support.lumerical.com/hc/en-us/articles/360036107914-Optical-N-Port-S-Parameter-SPAR-INTERCONNECT-Element#toc_5
    """

    assert isinstance(component, gf.Component)
    filepath = get_sparameters_path(
        component=component,
        dirpath=dirpath,
        layer_to_material=layer_to_material or layer_stack.get_layer_to_material(),
        layer_to_thickness_nm=layer_to_thickness_nm
        or layer_stack.get_layer_to_thickness_nm(),
    )
    numports = len(component.ports)
    assert (
        filepath.exists()
    ), f"Sparameters for {component.name} not found in {filepath}"
    assert numports > 1, f"number of ports = {numports} and needs to be > 1"
    return read_sparameters_lumerical(filepath=filepath, numports=numports)


def read_sparameters_pandas(
    component: Component,
    layer_to_material: Optional[Dict[Tuple[int, int], str]] = None,
    layer_to_thickness_nm: Optional[Dict[Tuple[int, int], int]] = None,
    dirpath: Path = gf.CONFIG["sp"],
) -> pd.DataFrame:
    filepath = get_sparameters_path(
        component=component,
        dirpath=dirpath,
        layer_to_material=layer_to_material or LAYER_STACK.get_layer_to_material(),
        layer_to_thickness_nm=layer_to_thickness_nm
        or LAYER_STACK.get_layer_to_thickness_nm(),
    )
    df = pd.read_csv(filepath.with_suffix(".csv"))
    df.index = df["wavelength_nm"]
    return df


if __name__ == "__main__":
    # test_read_sparameters_2port_straight()
    # test_read_sparameters_2port_bend()
    # test_read_sparameters_3port_mmi1x2()
    # test_read_sparameters_4port_mmi2x2()
    s = read_sparameters_component(gf.components.mmi2x2())
    # print(s[0])
    # print(s)
