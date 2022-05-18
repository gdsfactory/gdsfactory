import numpy as np

import gdsfactory as gf
from gdsfactory.simulation.simphony.circuit import component_to_circuit
from gdsfactory.simulation.simphony.get_transmission import get_transmission

mmi_name = "mmi1x2"
splitter = f"{mmi_name}_2p75_0p0"
combiner = f"{mmi_name}_68p45_0p0"


def test_circuit_transmission(data_regression, check: bool = True):
    component = gf.components.mzi(delta_length=10)
    circuit = component_to_circuit(component)

    for element in circuit._get_components():
        print(element)

    circuit._get_components()[0][0].rename("o1")
    circuit._get_components()[-1][-1].rename("o2")
    r = get_transmission(circuit.to_subcircuit(), num=3)
    s = np.round(r["s"], decimals=10).tolist()
    if check:
        data_regression.check(dict(w=r["wavelengths"].tolist(), s=s))
    return circuit


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # from gdsfactory.simulation.simphony import plot_circuit

    c = gf.c.mzi(delta_length=10)
    n = c.get_netlist()
    m = component_to_circuit(c)

    # for element in m._get_components():
    #     print(element)
    m.elements[splitter].pins["o1"] = "o1"
    m.elements[combiner].pins["o1"] = "o2"

    # plot_circuit(m)
    # plt.show()
    test_circuit_transmission(None, False)
