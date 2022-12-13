from __future__ import annotations

import gdsfactory as gf
from gdsfactory.simulation.simphony.model_from_gdsfactory import (
    GDSFactorySimphonyWrapper,
)


def mmi2x2(**kwargs):
    r"""Return 2x2 MultiModeInterferometer Sparameter model.

    Keyword Args:
        width: input and output straight width.
        width_taper: interface between input straights and mmi region.
        length_taper: into the mmi region.
        length_mmi: in x direction.
        width_mmi: in y direction.
        gap_mmi: (width_taper + gap between tapered wg)/2.
        taper: taper function.
        straight: straight function.
        with_bbox: box in bbox_layers and bbox_offsets to avoid DRC sharp edges.
        cross_section: spec.

    .. code::

                   length_mmi
                    <------>
                    ________
                   |        |
                __/          \__
            o2  __            __  o3
                  \          /_ _ _ _
                  |         | _ _ _ _| gap_mmi
                __/          \__
            o1  __            __  o4
                  \          /
                   |________|

                 <->
            length_taper

    .. plot::
      :include-source:

      import gdsfactory as gf
      c = gf.components.mmi2x2(length_mmi=15.45, width_mmi=2.1)
      c.plot()


    .. plot::
        :include-source:

        import gdsfactory.simulation.simphony as gs
        import gdsfactory.simulation.simphony.components as gc

        c = gc.mmi2x2()
        gs.plot_model(c)
    """
    return GDSFactorySimphonyWrapper(component=gf.components.mmi2x2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    wav = np.linspace(1520, 1570, 1024) * 1e-9
    f = 3e8 / wav
    c = mmi2x2()
    s = c.s_parameters(freqs=f)

    plt.plot(wav, np.abs(s[:, 1] ** 2))
    print(c.pins)
    plt.show()
