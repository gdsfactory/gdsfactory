from simphony.libraries import siepic

import gdsfactory as gf
from gdsfactory.simulation.simphony.components.gc import gc1550te


def add_gc(circuit, gc=gc1550te, cpi="o1", cpo="o2", gpi="port 1", gpo="port 2"):
    """Add input and output gratings.

    Args:
        circuit: needs to have `input` and `output` pins
        gc: grating coupler
        cpi: circuit pin input name
        cpo: circuit pin output name
        gpi: grating pin input name
        gpo: grating pin output name

    .. code::
                    _______
                   |       |
        gpi-> gpo--|cpi cpo|--gpo <-gpi
                   |_______|
    """
    gc_input = gf.call_if_func(gc)
    gc_input.rename_pins("o1", "o2")
    gc_output = gf.call_if_func(gc)
    gc_output.rename_pins("o1", "o2")

    circuit.pins[cpi].connect(gc_input)
    circuit.pins[cpo].connect(gc_output)

    return circuit


def add_gc_siepic(circuit, gc=siepic.GratingCoupler):
    """Add input and output gratings.

    Args:
        circuit: needs to have `o1` and `o2` pins
        gc: grating coupler
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
    plot_circuit(c2)
    plt.show()
