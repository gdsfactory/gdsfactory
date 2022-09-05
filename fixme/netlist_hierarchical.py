"""FIXME."""
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import sax
import gdsfactory as gf


def straight(wl=1.5, length=10.0, neff=2.4) -> sax.SDict:
    return sax.reciprocal({("o1", "o2"): jnp.exp(2j * jnp.pi * neff * length / wl)})


def mmi1x2() -> sax.SDict:
    """Returns an ideal 1x2 splitter."""
    return sax.reciprocal(
        {
            ("o1", "o2"): 0.5**0.5,
            ("o1", "o3"): 0.5**0.5,
        }
    )


def mmi2x2(*, coupling: float = 0.5) -> sax.SDict:
    """Returns an ideal 2x2 splitter.

    Args:
        coupling: power coupling coefficient.
    """
    kappa = coupling**0.5
    tau = (1 - coupling) ** 0.5
    return sax.reciprocal(
        {
            ("o1", "o4"): tau,
            ("o1", "o3"): 1j * kappa,
            ("o2", "o4"): 1j * kappa,
            ("o2", "o3"): tau,
        }
    )


def bend_euler(wl=1.5, length=20.0) -> sax.SDict:
    """Returns bend Sparameters with reduced transmission compared to a straight."""
    return {k: 0.99 * v for k, v in straight(wl=wl, length=length).items()}


def phase_shifter_heater(
    wl: float = 1.55,
    neff: float = 2.34,
    voltage: float = 0,
    length: float = 10,
    loss: float = 0.0,
) -> sax.SDict:
    """Returns simple phase shifter model.

    Args:
        wl: wavelength in um.
        neff: effective index.
        voltage: voltage per PI phase shift.
        length: in um.
        loss: in dB.
    """
    deltaphi = voltage * jnp.pi
    phase = 2 * jnp.pi * neff * length / wl + deltaphi
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    sdict = sax.reciprocal(
        {
            ("o1", "o2"): transmission,
        }
    )
    return sdict


models = {
    "bend_euler": bend_euler,
    "mmi1x2": mmi1x2,
    "mmi2x2": mmi2x2,
    "straight": straight,
    "taper": straight,
    "straight_heater_metal_undercut": phase_shifter_heater,
}


if __name__ == "__main__":
    c = gf.components.switch_tree(bend_s=None)
    n = c.get_netlist_recursive()

    netlist = c.get_netlist_recursive(exclude_port_types=("electrical", "placement"))
    mzi_circuit, _ = sax.circuit(netlist=netlist, models=models)
    S = mzi_circuit(wl=1.55)
    wl = np.linspace(1.5, 1.6, 256)
    S = mzi_circuit(wl=wl)

    plt.figure(figsize=(14, 4))
    plt.title("MZI")
    plt.plot(1e3 * wl, jnp.abs(S["o1", "o2"]) ** 2)
    plt.xlabel("λ [nm]")
    plt.ylabel("T")
    plt.grid(True)
    plt.show()
