from simphony.libraries import siepic

import gdsfactory as gf
from gdsfactory.simulation.simphony.components.gc import gc1550te


def add_gc(circuit, gc=gc1550te, ci="o1", co="o2", gi="port 1", go="port 2"):
    """Add input and output gratings.

    FIXME: does not work.

    Args:
        circuit: needs to have input ci and output co pins.
        gc: grating coupler.
        ci: circuit pin input name.
        co: circuit pin output name.
        gi: grating pin input name.
        go: grating pin output name.

    .. code::
                    _______
                   |       |
         gi-> gpo--|cpi cpo|--gpo <-gpi
                   |_______|

    """
    gc_input = gf.call_if_func(gc)
    gc_input.rename_pins(gi, go)
    gc_output = gf.call_if_func(gc)
    gc_output.rename_pins(gi, go)

    circuit.pins[ci].connect(gc_input)
    circuit.pins[co].connect(gc_output)
    return circuit


def add_gc_siepic(circuit, gc=siepic.GratingCoupler):
    """Add input and output gratings.

    Args:
        circuit: needs to have `o1` and `o2` pins.
        gc: grating coupler.

    """
    gci = gco = gc
    gci["n1"].connect(gco["n1"])
    gci["n2"].rename("o1")
    gco["n1"].rename("o2")

    return gci.circuit.to_subcircuit()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from gdsfactory.simulation.simphony.components.mzi import mzi
    from gdsfactory.simulation.simphony.plot_circuit import plot_circuit

    c1 = mzi()
    c2 = add_gc(c1)
    plot_circuit(c2, pin_in="o1")
    plt.show()
