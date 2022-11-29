from __future__ import annotations

from simphony import Model

import gdsfactory as gf
from gdsfactory.simulation.simphony.model_from_gdsfactory import (
    GDSFactorySimphonyWrapper,
)


def coupler_ring_fdtd(
    factory=gf.c.coupler_ring, width=0.5, length_x=4.0, gap=0.2, radius=5
) -> Model:
    r"""Return half ring model based on Lumerical 3D FDTD simulations.

    Args:
        c: gdsfactory component
        width:0.5
        gap: 0.2
        length_x: 4
        radius: 5

    .. code::

           N0            N1
           |             |
            \           /
             \         /
           ---=========---
        W0    length_x    E0


    """
    coupler = (
        factory(width=width, length_x=length_x, gap=gap, radius=radius)
        if callable(factory)
        else factory
    )
    return GDSFactorySimphonyWrapper(component=coupler)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    import gdsfactory.simulation.simphony as gs

    wav = np.linspace(1520, 1570, 1024) * 1e-9
    c = coupler_ring_fdtd()
    wavelengths = np.linspace(1.5, 1.6) * 1e-6
    gs.plot_model(c, wavelengths=wavelengths)
    plt.show()

    # f = 3e8 / wav
    # s = c.s_parameters(freqs=f)
    # plt.plot(wav, np.abs(s[:, 1] ** 2))
    # print(c.pins)
    # plt.show()
