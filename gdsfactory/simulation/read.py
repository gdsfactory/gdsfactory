import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.simulation.get_sparameters_path import get_sparameters_path
from gdsfactory.tech import LAYER_STACK, LayerStack


def get_ports(line: str) -> Tuple[str, str]:
    """Returns 2 port labels strings from interconnect file."""
    line = line.replace('"', "")
    line = line.replace("(", "")
    line_fields = line.split(",")
    port1 = line_fields[0]
    port2 = line_fields[3]
    return port1, port2


def _read_sparameters_file(
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


def read_sparameters_lumerical(
    component: Optional[Component] = None,
    layer_stack: LayerStack = LAYER_STACK,
    filepath: Optional[str] = None,
    numports: Optional[int] = None,
    dirpath: Path = gf.CONFIG["sparameters"],
) -> Tuple[List[str], np.array, np.ndarray]:
    r"""Returns Sparameters from Lumerical interconnect .DAT file.

    Args:
        component: Component
        layer_stack:
        filepath:
        numports: number of ports
        dirpath: path where to look for the Sparameters

    Returns:
        port_names: list of port labels
        F: frequency 1d np.array
        S: Sparameters np.ndarray matrix


    the Sparameters file have Lumerical format
    https://support.lumerical.com/hc/en-us/articles/360036107914-Optical-N-Port-S-Parameter-SPAR-INTERCONNECT-Element#toc_5
    """

    if component is None and filepath is None:
        raise ValueError("You need to define the filepath or the component")

    if filepath and numports is None:
        raise ValueError("You need to define numports")

    filepath = filepath or get_sparameters_path(
        component=component,
        dirpath=dirpath,
        layer_to_material=layer_stack.get_layer_to_material(),
        layer_to_thickness=layer_stack.get_layer_to_thickness(),
    )
    numports = numports or len(component.ports)
    assert (
        filepath.exists()
    ), f"Sparameters for {component.name} not found in {filepath}"
    assert numports > 1, f"number of ports = {numports} and needs to be > 1"
    return _read_sparameters_file(filepath=filepath, numports=numports)


def read_sparameters_pandas(
    component: Component,
    layer_stack: LayerStack = LAYER_STACK,
    dirpath: Path = gf.CONFIG["sparameters"],
) -> pd.DataFrame:
    filepath = get_sparameters_path(
        component=component,
        dirpath=dirpath,
        layer_to_material=layer_stack.get_layer_to_material(),
        layer_to_thickness=layer_stack.get_layer_to_thickness(),
    )
    df = pd.read_csv(filepath.with_suffix(".csv"))
    df.index = df["wavelength_nm"]
    return df


if __name__ == "__main__":
    s = read_sparameters_pandas(gf.components.mmi2x2())
