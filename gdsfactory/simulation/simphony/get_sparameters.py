import numpy as np
from simphony.tools import wl2freq


def get_sparameters(model, wavelengths=None):
    """Return wavelengths, Sparameters for a model.

    Args:
        model: model function of Model
        wavelengths: (m)
    """
    wavelengths = wavelengths or np.linspace(1520, 1570, 3) * 1e-9
    f = wl2freq(wavelengths)
    model = model() if callable(model) else model
    s = model.s_parameters(freq=f)
    return wavelengths, s


if __name__ == "__main__":
    import gdslib as gl

    w, sp = get_sparameters(gl.simphony.components.straight)
    print(abs(sp))
