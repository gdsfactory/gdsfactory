import numpy as np
import pytest

from gdsfactory.simulation.simphony.circuit import get_transmission
from gdsfactory.simulation.simphony.components import (
    circuit_factory,
    circuit_names,
    component_factory,
    component_names,
)


@pytest.mark.parametrize("component_type", component_names)
def test_elements(component_type, data_regression):
    c = component_factory[component_type]()
    wav = np.linspace(1520, 1570, 3) * 1e-9
    f = 3e8 / wav
    s = c.s_parameters(freq=f)
    _, rows, cols = np.shape(s)
    sdict = {
        f"S{i+1}{j+1}": np.round(np.abs(s[:, i, j]), decimals=3).tolist()
        for i in range(rows)
        for j in range(cols)
    }
    data_regression.check(sdict)


@pytest.mark.parametrize("component_type", circuit_names)
def test_circuits(component_type, data_regression):
    c = circuit_factory[component_type]()
    r = get_transmission(c, num=3)
    s = np.round(r["s"], decimals=3).tolist()
    data_regression.check(dict(w=r["wavelength_nm"].tolist(), s=s))
