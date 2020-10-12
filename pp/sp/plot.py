import numpy as np
import pp
from pp.sp.write import write


def plot(
    component_or_results_dict,
    logscale=True,
    keys=None,
    height_nm=220,
    dirpath=pp.CONFIG["sp"],
    **kwargs,
):
    """ plots Sparameters

    Args:
        component_or_results_dict:
        logscale: plots 20*log10(results)
        keys: list of keys to plot
        height_nm: nm height
        dirpath: where to store the simulations
        **kwargs: plotting kwargs

    """
    import matplotlib.pyplot as plt

    r = component_or_results_dict
    if isinstance(r, pp.Component):
        r = write(component=r, height_nm=height_nm, dirpath=dirpath)
    w = r["wavelength_nm"]

    if keys:
        keys = [key for key in keys if key in r.keys()]
    else:
        keys = [key for key in r.keys() if key.startswith("S") and key.endswith("m")]

    for key in keys:
        if logscale:
            y = 20 * np.log10(r[key])
        else:
            y = r[key]

        plt.plot(w, y, label=key[:-1], **kwargs)
    plt.legend()
    plt.xlabel("wavelength (nm)")
    if logscale:
        plt.ylabel("Transmission (dB)")
    else:
        plt.ylabel("Transmission")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    remove_layers = []
    layer2nm = {(1, 0): 220}

    # r = write(component=pp.c.waveguide(), layer2nm=layer2nm)
    # r = write(component=pp.c.mmi2x2(), layer2nm=layer2nm)
    # r = write(component=pp.c.mmi1x2(), layer2nm=layer2nm)
    r = write(component=pp.c.coupler(), layer2nm=layer2nm)
    # r = write(component=pp.c.bend_circular(), layer2nm=layer2nm)
    plot(r, logscale=True)
    plt.show()
