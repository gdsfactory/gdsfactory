import re
import numpy as np
import pp


def load(
    component=None, filepath=None, dirpath=pp.CONFIG["sp"], numports=None, height_nm=220
):
    """ Load Sparameters from Lumerical interconnect export file

    Args:
        component: instance
        filepath: Sparameters filepath (interconnect format)
        dirpath: path where to look for the Sparameters
        height_nm: height (nm)

    Returns [port_names, F, S]
        port_names: list of strings
        F: frequency 1d np.array
        S: Sparameters np.ndarray matrix

    inspired in https://github.com/BYUCamachoLab/simphony
    the Sparameters file have Lumerical format
    https://support.lumerical.com/hc/en-us/articles/360036107914-Optical-N-Port-S-Parameter-SPAR-INTERCONNECT-Element#toc_5
    """
    if filepath is None:
        assert isinstance(component, pp.Component)
        filepath = component.get_sparameters_path(dirpath=dirpath, height_nm=height_nm)
        numports = len(component.ports)
    assert filepath.exists(), f"Sparameters for {component} not found in {filepath}"
    assert numports > 1, f"number of ports = {numports} and needs to be > 1"

    F = []
    S = []
    port_names = []

    with open(filepath, "r") as fid:
        for i in range(numports):
            port_line = fid.readline()
            m = re.search('\[".*",', port_line)
            if m:
                port = m.group(0)
                port_names.append(port[2:-2])
        line = fid.readline()
        line = fid.readline()
        numrows = int(tuple(line[1:-2].split(","))[0])
        S = np.zeros((numrows, numports, numports), dtype="complex128")
        r = m = n = 0
        for line in fid:
            if line[0] == "(":
                continue
            data = line.split()
            data = list(map(float, data))
            if m == 0 and n == 0:
                F.append(data[0])
            S[r, m, n] = data[1] * np.exp(1j * data[2])
            r += 1
            if r == numrows:
                r = 0
                m += 1
                if m == numports:
                    m = 0
                    n += 1
                    if n == numports:
                        break
    return (port_names, F, S)


if __name__ == "__main__":
    s = load(pp.c.mmi2x2())
    print(s[0])
    # print(s)
