import numpy as np
import pp


def load(
    component, dirpath=pp.CONFIG["gdslib"],
):
    """ returns 2 np.ndarray [frequency, s-parameters]
    inspired in https://github.com/BYUCamachoLab/simphony
    """
    assert isinstance(component, pp.Component)

    output_folder = dirpath / component.name
    filepath = output_folder / component.name
    filepath_sp = filepath.with_suffix(".dat")
    assert filepath_sp.exists()

    numports = len(component.ports)

    F = []
    S = []
    with open(filepath_sp, "r") as fid:
        for i in range(numports):
            fid.readline()
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
    return (F, S)


if __name__ == "__main__":
    s = load(pp.c.waveguide())
    print(s)
