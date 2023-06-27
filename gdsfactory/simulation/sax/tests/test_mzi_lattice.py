"""Test hierarchical circuit simulations."""
from __future__ import annotations

from typing import List

import jax.numpy as jnp
import numpy as np
import sax

import gdsfactory as gf


@gf.cell
def mzis():
    c = gf.Component()
    c1 = c << gf.components.mzi(delta_length=10)
    c2 = c << gf.components.mzi(delta_length=10)
    c2.connect("o1", c1.ports["o2"])

    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c2.ports["o2"])
    return c


def straight(wl=1.5, length=10.0, neff=2.4) -> sax.SDict:
    """Straight model."""
    return sax.reciprocal({("o1", "o2"): jnp.exp(2j * jnp.pi * neff * length / wl)})


def mmi1x2():
    """Assumes a perfect 1x2 splitter."""
    return sax.reciprocal(
        {
            ("o1", "o2"): 0.5**0.5,
            ("o1", "o3"): 0.5**0.5,
        }
    )


def bend_euler(wl=1.5, length=20.0):
    """Assumes reduced transmission for the euler bend compared to a straight."""
    return {k: 0.99 * v for k, v in straight(wl=wl, length=length).items()}


models = {
    "bend_euler": bend_euler,
    "mmi1x2": mmi1x2,
    "straight": straight,
}


def module(S) -> List[float]:
    """rounds to 3 decimals and converts numpy to lists for serialization."""
    for k, v in S.items():
        S[k] = [float(i) for i in np.round(np.abs(v) ** 2, 3)]
    return S


def test_mzi_lattice(data_regression, check: bool = True) -> None:
    c = mzis()
    netlist = c.get_netlist_recursive()
    circuit, _ = sax.circuit(netlist=netlist, models=models)
    c.show(show_ports=True)
    wl = np.linspace(1.5, 1.6, 3)
    S = circuit(wl=wl)
    S = module(S)
    if check:
        d = dict(S21=S["o1", "o2"], S11=S["o1", "o1"])
        data_regression.check(d)


if __name__ == "__main__":
    c = mzis()
    # netlist = c.get_netlist()
    netlist = c.get_netlist_recursive()
    circuit, _ = sax.circuit(netlist=netlist, models=models)
    c.show(show_ports=True)
    wl = np.linspace(1.5, 1.6, 3)
    S = circuit(wl=wl)
    S = module(S)
    d = dict(S21=S["o1", "o2"], S11=S["o1", "o1"])

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(14, 4))
    # plt.title("MZI")
    # plt.plot(1e3 * wl, jnp.abs(S["o1", "o2"]) ** 2)
    # plt.xlabel("λ [nm]")
    # plt.ylabel("T")
    # plt.grid(True)
    # plt.show()
