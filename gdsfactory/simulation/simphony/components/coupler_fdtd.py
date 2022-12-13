from __future__ import annotations

import gdsfactory as gf
from gdsfactory.simulation.simphony.model_from_gdsfactory import (
    GDSFactorySimphonyWrapper,
)


def coupler_fdtd(c=gf.c.coupler, width=0.5, length=20, gap=0.224):
    r"""Coupler based on Lumerical 3D FDTD simulations.

    Args:
        c: Coupler function
        width:0.5
        length: 4
        gap: 0.2

    .. code::

       W1 __             __ E1
            \           /
             \         /
              ========= gap
             /          \
           _/            \_
        W0      length    E0


    """
    if callable(c):
        c = c(width=width, length=length, gap=gap)
    return GDSFactorySimphonyWrapper(component=c)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    wav = np.linspace(1520, 1570, 1024) * 1e-9
    f = 3e8 / wav
    c = gf.c.coupler(length=20, gap=0.224)
    m = coupler_fdtd(c=c)
    s = m.s_parameters(freqs=f)

    plt.plot(wav, np.abs(s[:, 1] ** 2))
    print(m.pins)
    plt.show()
