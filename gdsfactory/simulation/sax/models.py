import jax.numpy as jnp
from sax.typing_ import SDict
from sax.utils import reciprocal


def straight(
    *,
    wl: float = 1.55,
    wl0: float = 1.55,
    neff: float = 2.34,
    ng: float = 3.4,
    length: float = 10.0,
    loss: float = 0.0
) -> SDict:
    """Simple straight waveguide model.

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


def coupler(*, coupling: float = 0.5) -> SDict:
    r"""Simple coupler model for a single wavelength.

    Args:
        coupler: coupling coefficient

    .. code::

         o2 ________                           ______o3
                    \                         /
                     \        length         /
                      ======================= gap
                     /                       \
            ________/                         \_______
         o1                                          o4

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
