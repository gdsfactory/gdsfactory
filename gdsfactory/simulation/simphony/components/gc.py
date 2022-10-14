from gdsfactory.config import sparameters_path
from gdsfactory.simulation.simphony.model_from_sparameters import SimphonyFromFile


def gc1550te(filepath=sparameters_path / "gc2dte" / "gc1550.dat", numports=2):
    """Returns Sparameter model for 1550nm TE grating_coupler.

    .. plot::
        :include-source:

        import gdsfactory.simulation.simphony as gs
        import gdsfactory.simulation.simphony.components as gc

        c = gc.gc1550te()
        gs.plot_model(c)

    """
    return SimphonyFromFile(numports=numports).model_from_filepath(filepath=filepath)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    wav = np.linspace(1520, 1570, 1024) * 1e-9
    f = 3e8 / wav
    c = gc1550te()
    s = c.s_parameters(freqs=f)

    plt.plot(wav, np.abs(s[:, 1] ** 2))
    print(c.pins)
    # plt.legend()
    # plt.show()
