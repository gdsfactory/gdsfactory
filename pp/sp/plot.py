import matplotlib.pyplot as plt
import numpy as np
import pp
from pp.sp.write import write


def plot(r, logscale=True):
    """ plots Sparameters
    """
    w = r["wavelength_nm"]
    for key in r.keys():
        if key.startswith("S") and key.endswith("m"):
            if logscale:
                y = 20 * np.log10(r[key])
            else:
                y = r[key]

            plt.plot(w, y, label=key[:-1])
    plt.legend()
    plt.xlabel("wavelength (nm)")


if __name__ == "__main__":
    remove_layers = []
    layer2nm = {(1, 0): 220}

    r = write(component=pp.c.waveguide(), layer2nm=layer2nm)
    # r = write(component=pp.c.mmi2x2(), layer2nm=layer2nm)
    # r = write(component=pp.c.mmi1x2(), layer2nm=layer2nm)
    # r = write(component=pp.c.coupler(), layer2nm=layer2nm)
    # r = write(component=pp.c.bend_circular(), layer2nm=layer2nm)
    plot(r, logscale=True)
    plt.show()
