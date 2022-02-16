import jax.numpy as jnp
from sax.typing_ import SDict
from sax.utils import reciprocal

nm = 1e-3


def straight(
    *,
    wl: float = 1.55,
    wl0: float = 1.55,
    neff: float = 2.34,
    ng: float = 3.4,
    length: float = 10.0,
    loss: float = 0.0,
) -> SDict:
    """dispersive straight waveguide model.

    Args:
        wl: wavelength
        wl0: center wavelength
        neff: effective index
        ng: group index
        length: um
        loss: in dB/um

    .. code::

        o1 -------------- o2
                length
    """
    dwl = wl - wl0
    dneff_dwl = (ng - neff) / wl0
    neff = neff - dwl * dneff_dwl
    phase = 2 * jnp.pi * neff * length / wl
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    sdict = reciprocal(
        {
            ("o1", "o2"): transmission,
        }
    )
    return sdict


def attenuator(*, loss: float = 0.0) -> SDict:
    """attenuator model.

    Args:
        loss: in dB

    .. code::

        o1 -------------- o2
                loss
    """
    transmission = jnp.asarray(10 ** (-loss / 20), dtype=complex)
    sdict = reciprocal(
        {
            ("o1", "o2"): transmission,
        }
    )
    return sdict


def grating_coupler(
    *,
    wl: float = 1.55,
    wl0: float = 1.55,
    loss: float = 0.0,
    reflection: float = 0.0,
    reflection_fiber: float = 0.0,
    bandwidth: float = 40 * nm,
) -> SDict:
    """grating_coupler model.

    Args:
        wl0: center wavelength
        loss: in dB
        reflection: from waveguide side.
        reflection_fiber: from fiber side.
        bandwidth: 1dB bandwidth (um)

    .. code::

                      fiber o2

                   /  /  /  /
                  /  /  /  /

                _|-|_|-|_|-|___
            o1  ______________|

    """
    amplitude = jnp.asarray(10 ** (-loss / 20), dtype=complex)
    sigma = bandwidth / 2 * jnp.sqrt(10 / (2 * jnp.log(10)))
    transmission = amplitude * jnp.exp(-((wl - wl0) ** 2) / (2 * sigma ** 2))
    sdict = reciprocal(
        {
            ("o1", "o1"): reflection * jnp.ones_like(transmission),
            ("o1", "o2"): transmission,
            ("o2", "o1"): transmission,
            ("o2", "o2"): reflection_fiber * jnp.ones_like(transmission),
        }
    )
    return sdict


def coupler(*, coupling: float = 0.5) -> SDict:
    r"""coupler model for a single wavelength.

    Args:
        coupling: power coupling coefficient.

    .. code::

         o2 ________                           ______o3
                    \                         /
                     \        length         /
                      ======================= gap
                     /                       \
            ________/                         \_______
         o1                                          o4

                   ---------------------------> 1 - coupling
    """
    kappa = coupling ** 0.5
    tau = (1 - coupling) ** 0.5
    sdict = reciprocal(
        {
            ("o1", "o4"): tau,
            ("o1", "o3"): 1j * kappa,
            ("o2", "o4"): 1j * kappa,
            ("o2", "o3"): tau,
        }
    )
    return sdict


if __name__ == "__main__":
    import gdsfactory.simulation.sax as gs

    gs.plot_model(grating_coupler)
    # gs.plot_model(coupler)
