"""FIXME.

Where is the light going?
"""
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


def bend_euler(wl: float = 1.5, length: float = 20.0, loss: float = 50e-3) -> sax.SDict:
    """Returns bend Sparameters."""
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    return {k: amplitude * v for k, v in straight(wl=wl, length=length).items()}


def phase_shifter(
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
    return sax.reciprocal(
        {
            ("o1", "o2"): transmission,
        }
    )


models = {
    "bend_euler": bend_euler,
    "mmi1x2": mmi1x2,
    "mmi2x2": mmi2x2,
    "straight": straight,
    "taper": straight,
    "straight_heater_metal_undercut": phase_shifter,
    "compass": sax.models.passthru(10),
    "via": sax.models.passthru(10),
}


if __name__ == "__main__":
    c = gf.components.switch_tree(bend_s=None)
    c.show(show_ports=True)
    n = netlist = c.get_netlist_recursive(
        exclude_port_types=("electrical", "placement")
    )
    # netlist.pop(list(n.keys())[0])
    mzi_circuit, _ = sax.circuit(netlist=netlist, models=models)
    S = mzi_circuit(wl=1.55)
    wl = np.linspace(1.5, 1.6, 256)
    S = mzi_circuit(wl=wl)

    plt.figure(figsize=(14, 4))
    plt.title("MZI")

    plt.plot(1e3 * wl, jnp.abs(S["o1_0_0", "o2_1_9"]) ** 2)
    plt.plot(1e3 * wl, jnp.abs(S["o1_0_0", "o3_1_6"]) ** 2)
    plt.plot(1e3 * wl, jnp.abs(S["o1_0_0", "o2_1_5"]) ** 2)
    plt.plot(1e3 * wl, jnp.abs(S["o1_0_0", "o3_1_10"]) ** 2)
    plt.xlabel("λ [nm]")
    plt.ylabel("T")
    plt.grid(True)
    plt.show()
