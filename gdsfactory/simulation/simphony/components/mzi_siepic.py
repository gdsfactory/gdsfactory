from __future__ import annotations

from simphony.libraries import siepic

from gdsfactory.simulation.simphony.components.mmi1x2 import mmi1x2


def mzi(L0=1, DL=100.0, L2=10.0, y_model_factory=mmi1x2, wg=siepic.Waveguide):
    """Mzi circuit model.

    Args:
        L0 (um): vertical length for both and top arms
        DL (um): bottom arm extra length, delta_length = 2*DL
        L2 (um): L_top horizontal length

    Return: mzi circuit model

    .. code::

               __L2__
               |      |
               L0     L0r
               |      |
     splitter==|      |==recombiner
               |      |
               L0     L0r
               |      |
               DL     DL
               |      |
               |__L2__|


    .. plot::
      :include-source:

      import gdsfactory as gf

      c = gf.c.mzi(L0=0.1, DL=0, L2=10)
      gf.plotgds(c)


    .. plot::
        :include-source:

        import gdsfactory.simulation.simphony as gs
        import gdsfactory.simulation.simphony.components as gc

        c = gc.mzi()
        gs.plot_circuit(c)

    """
    y_splitter = y_model_factory() if callable(y_model_factory) else y_model_factory
    y_recombiner = y_model_factory() if callable(y_model_factory) else y_model_factory
    wg_long = wg(length=(2 * L0 + 2 * DL + L2) * 1e-6)
    wg_short = wg(length=(2 * L0 + L2) * 1e-6)

    y_recombiner.pins[0].rename("o2")

    y_splitter[1].connect(wg_long)
    y_splitter[2].connect(wg_short)
    y_recombiner.multiconnect(None, wg_long, wg_short)

    return y_splitter.circuit.to_subcircuit("mzi")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from gdsfactory.simulation.simphony import plot_circuit

    c = mzi()
    plot_circuit(c)
    plt.show()
