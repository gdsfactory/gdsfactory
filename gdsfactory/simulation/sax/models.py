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
    """Dispersive straight waveguide model.

    based on sax.models

    Args:
        wl: wavelength.
        wl0: center wavelength.
        neff: effective index.
        ng: group index.
        length: um.
        loss: in dB/um.

    .. code::

        o1 -------------- o2
                length

    """
    dwl = wl - wl0
    dneff_dwl = (ng - neff) / wl0
    neff -= dwl * dneff_dwl
    phase = 2 * jnp.pi * neff * length / wl
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    return reciprocal(
        {
            ("o1", "o2"): transmission,
        }
    )


def bend(wl: float = 1.5, length: float = 20.0, loss: float = 0.0) -> SDict:
    """Returns bend Sparameters."""
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    return {k: amplitude * v for k, v in straight(wl=wl, length=length).items()}


def attenuator(*, loss: float = 0.0) -> SDict:
    """Attenuator model.

    based on sax.models

    Args:
        loss: in dB.

    .. code::

        o1 -------------- o2
                loss

    """
    transmission = jnp.asarray(10 ** (-loss / 20), dtype=complex)
    return reciprocal(
        {
            ("o1", "o2"): transmission,
        }
    )


def phase_shifter(
    wl: float = 1.55,
    neff: float = 2.34,
    voltage: float = 0,
    length: float = 10,
    loss: float = 0.0,
) -> SDict:
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
    return reciprocal(
        {
            ("o1", "o2"): transmission,
        }
    )


def grating_coupler(
    *,
    wl: float = 1.55,
    wl0: float = 1.55,
    loss: float = 0.0,
    reflection: float = 0.0,
    reflection_fiber: float = 0.0,
    bandwidth: float = 40 * nm,
) -> SDict:
    """Grating_coupler model.

    equation adapted from photontorch grating coupler
    https://github.com/flaport/photontorch/blob/master/photontorch/components/gratingcouplers.py

    Args:
        wl0: center wavelength.
        loss: in dB.
        reflection: from waveguide side.
        reflection_fiber: from fiber side.
        bandwidth: 3dB bandwidth (um).

    .. code::

                      fiber o2

                   /  /  /  /
                  /  /  /  /

                _|-|_|-|_|-|___
            o1  ______________|

    """
    amplitude = jnp.asarray(10 ** (-loss / 20), dtype=complex)
    sigma = bandwidth / (2 * jnp.sqrt(2 * jnp.log(2)))
    transmission = amplitude * jnp.exp(-((wl - wl0) ** 2) / (2 * sigma**2))
    return reciprocal(
        {
            ("o1", "o1"): reflection * jnp.ones_like(transmission),
            ("o1", "o2"): transmission,
            ("o2", "o1"): transmission,
            ("o2", "o2"): reflection_fiber * jnp.ones_like(transmission),
        }
    )


def coupler(
    *,
    wl: float = 1.55,
    wl0: float = 1.55,
    length: float = 0.0,
    coupling0: float = 0.2,
    dk1: float = 1.2435,
    dk2: float = 5.3022,
    dn: float = 0.02,
    dn1: float = 0.1169,
    dn2: float = 0.4821,
) -> SDict:
    r"""Dispersive coupler model.

    equations adapted from photontorch.
    https://github.com/flaport/photontorch/blob/master/photontorch/components/directionalcouplers.py

    kappa = coupling0 + coupling

    Args:
        wl: wavelength (um).
        wl0: center wavelength (um).
        length: coupling length (um).
        coupling0: bend region coupling coefficient from FDTD simulations.
        dk1: first derivative of coupling0 vs wavelength.
        dk2: second derivative of coupling vs wavelength.
        dn: effective index difference between even and odd modes.
        dn1: first derivative of effective index difference vs wavelength.
        dn2: second derivative of effective index difference vs wavelength.

    .. code::

          coupling0/2        coupling        coupling0/2
        <-------------><--------------------><---------->
         o2 ________                           _______o3
                    \                         /
                     \        length         /
                      =======================
                     /                       \
            ________/                         \________
         o1                                           o4

                      ------------------------> K (coupled power)
                     /
                    / K
           -----------------------------------> T = 1 - K (transmitted power)
    """
    dwl = wl - wl0
    dn = dn + dn1 * dwl + 0.5 * dn2 * dwl**2
    kappa0 = coupling0 + dk1 * dwl + 0.5 * dk2 * dwl**2
    kappa1 = jnp.pi * dn / wl

    tau = jnp.cos(kappa0 + kappa1 * length)
    kappa = -jnp.sin(kappa0 + kappa1 * length)
    return reciprocal(
        {
            ("o1", "o4"): tau,
            ("o1", "o3"): 1j * kappa,
            ("o2", "o4"): 1j * kappa,
            ("o2", "o3"): tau,
        }
    )


def coupler_single_wavelength(*, coupling: float = 0.5) -> SDict:
    r"""Coupler model for a single wavelength.

    Based on sax.models.

    Args:
        coupling: power coupling coefficient.

    .. code::

         o2 ________                           ______o3
                    \                         /
                     \        length         /
                      =======================
                     /                       \
            ________/                         \_______
         o1                                          o4

    """
    kappa = coupling**0.5
    tau = (1 - coupling) ** 0.5
    return reciprocal(
        {
            ("o1", "o4"): tau,
            ("o1", "o3"): 1j * kappa,
            ("o2", "o4"): 1j * kappa,
            ("o2", "o3"): tau,
        }
    )


def mmi1x2() -> SDict:
    """Returns an ideal 1x2 splitter."""
    return reciprocal(
        {
            ("o1", "o2"): 0.5**0.5,
            ("o1", "o3"): 0.5**0.5,
        }
    )


def mmi2x2(*, coupling: float = 0.5) -> SDict:
    """Returns an ideal 2x2 splitter.

    Args:
        coupling: power coupling coefficient.
    """
    kappa = coupling**0.5
    tau = (1 - coupling) ** 0.5
    return reciprocal(
        {
            ("o1", "o4"): tau,
            ("o1", "o3"): 1j * kappa,
            ("o2", "o4"): 1j * kappa,
            ("o2", "o3"): tau,
        }
    )


models = dict(
    straight=straight,
    bend_euler=bend,
    mmi1x2=mmi1x2,
    mmi2x2=mmi2x2,
    attenuator=attenuator,
    taper=straight,
    phase_shifter=phase_shifter,
    grating_coupler=grating_coupler,
    coupler=coupler,
)


if __name__ == "__main__":
    import gdsfactory.simulation.sax as gs

    gs.plot_model(grating_coupler)
    # gs.plot_model(coupler)
