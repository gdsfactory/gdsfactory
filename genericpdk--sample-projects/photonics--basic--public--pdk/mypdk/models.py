"""SAX models for Sparameter circuit simulations."""

import inspect
from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import sax
import sax.models as sm
from numpy.typing import NDArray

sax.set_port_naming_strategy("optical")

nm = 1e-3

FloatArray = NDArray[jnp.floating]
Float = float | FloatArray

################
# Straights
################

straight_strip = partial(
    sm.straight,
    length=10.0,
    loss_dB_cm=3.0,
    wl0=1.55,
    neff=2.38,
    ng=4.30,
)

straight_rib = partial(
    sm.straight,
    length=10.0,
    loss_dB_cm=3.0,
    wl0=1.55,
    neff=2.38,
    ng=4.30,
)


def straight(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
    cross_section: str = "strip",
) -> sax.SDict:
    """Straight waveguide model."""
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "strip": straight_strip,
        "rib": straight_rib,
    }
    f = fs[cross_section]
    return f(
        wl=wl,  # type: ignore
        length=length,
        loss_dB_cm=loss_dB_cm,
    )


################
# Bends
################


def wire_corner(*, wl: Float = 1.55) -> sax.SDict:
    """Wire corner model."""
    wl = jnp.asarray(wl)  # type: ignore
    zero = jnp.zeros_like(wl)
    return {"e1": zero, "e2": zero}  # type: ignore


def bend_s(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm=3.0,
    cross_section="strip",
) -> sax.SDict:
    """Bend S model."""
    # NOTE: it is assumed that `bend_s` exposes it's length in its info dictionary!
    return straight(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


def bend_euler(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3,
    cross_section="strip",
) -> sax.SDict:
    """Euler bend model."""
    # NOTE: it is assumed that `bend_euler` exposes it's length in its info dictionary!
    return straight(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


bend_euler_strip = partial(bend_euler, cross_section="strip")
bend_euler_rib = partial(bend_euler, cross_section="rib")


################
# Transitions
################


def taper(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 0.0,
    cross_section="strip",
) -> sax.SDict:
    """Taper model."""
    # NOTE: it is assumed that `taper` exposes it's length in its info dictionary!
    # TODO: take width1 and width2 into account.
    return straight(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


taper_rib = partial(taper, cross_section="rib", length=10.0)


def taper_strip_to_ridge(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 0.0,
    cross_section="strip",
) -> sax.SDict:
    """Taper strip to ridge model."""
    # NOTE: it is assumed that `taper_strip_to_ridge` exposes it's length in its info dictionary!
    # TODO: take w_slab1 and w_slab2 into account.
    return straight(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


trans_rib10 = partial(taper_strip_to_ridge, length=10.0)
trans_rib20 = partial(taper_strip_to_ridge, length=20.0)
trans_rib50 = partial(taper_strip_to_ridge, length=50.0)

################
# MMIs
################

mmi1x2_strip = partial(sm.mmi1x2, wl0=1.55, fwhm=0.2)
mmi1x2_rib = mmi1x2_strip


def mmi1x2(
    wl: Float = 1.55,
    loss_dB: Float = 0.3,
    cross_section="strip",
) -> sax.SDict:
    """MMI 1x2 model."""
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "strip": mmi1x2_strip,
        "rib": mmi1x2_rib,
    }
    f = fs[cross_section]
    return f(
        wl=wl,
        loss_dB=loss_dB,
    )


mmi2x2_strip = partial(sm.mmi2x2, wl0=1.55, fwhm=0.2)
mmi2x2_rib = mmi2x2_strip


def mmi2x2(
    wl: Float = 1.55,
    loss_dB: Float = 0.3,
    cross_section="strip",
) -> sax.SDict:
    """MMI 2x2 model."""
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "strip": mmi2x2_strip,
        "rib": mmi2x2_rib,
    }
    f = fs[cross_section]
    return f(
        wl=wl,
        loss_dB=loss_dB,
    )


##############################
# Evanescent couplers
##############################

coupler_strip = partial(sm.coupler, wl0=1.55)
coupler_rib = coupler_strip
coupler_ring = partial(coupler_strip, wl0=1.55)


def coupler(
    wl: Float = 1.55,
    length: float = 10.0,
    cross_section="strip",
) -> sax.SDict:
    """Evanescent coupler model."""
    # TODO: take more coupler arguments into account
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "strip": coupler_strip,
        "rib": coupler_rib,
    }
    f = fs[cross_section]
    return f(
        wl=wl,
        length=length,
    )


##############################
# grating couplers Rectangular
##############################

grating_coupler_rectangular_strip = partial(
    sm.grating_coupler, loss=6, bandwidth=35 * nm, wl=1.55
)
grating_coupler_rectangular_rib = grating_coupler_rectangular_strip


def grating_coupler_rectangular(
    wl: Float = 1.55,
    cross_section="strip",
) -> sax.SDict:
    """Grating coupler rectangular model."""
    # TODO: take more grating_coupler_rectangular arguments into account
    wl = jnp.asarray(wl)  # type: ignore
    fs = {
        "strip": grating_coupler_rectangular_strip,
        "rib": grating_coupler_rectangular_rib,
    }
    f = fs[cross_section]
    return f(wl=wl)  # type: ignore


##############################
# grating couplers Elliptical
##############################

grating_coupler_elliptical = partial(
    sm.grating_coupler, loss=6, bandwidth=35 * nm, wl=1.55
)

################
# Imported
################


def heater() -> sax.SDict:
    """Heater model."""
    raise NotImplementedError("No model for 'heater'")


def straight_heater_metal(
    wl: float = 1.55,
    neff: float = 2.34,
    voltage: float = 0,
    vpi: float = 1.0,  # Voltage required for π-phase shift
    length: float = 10,
    loss: float = 0.0,
) -> sax.SDict:
    """Returns simple phase shifter model.

    Args:
        wl: wavelength.
        neff: effective index.
        voltage: applied voltage.
        vpi: voltage required for a π-phase shift.
        length: length.
        loss: loss.
    """
    # Calculate additional phase shift due to applied voltage.
    deltaphi = (voltage / vpi) * jnp.pi
    phase = 2 * jnp.pi * neff * length / wl + deltaphi
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    return sax.reciprocal(
        {
            ("o1", "o2"): transmission,
            ("l_e1", "r_e1"): 0.0,
            ("l_e2", "r_e2"): 0.0,
            ("l_e3", "r_e3"): 0.0,
            ("l_e4", "r_e4"): 0.0,
        }
    )


crossing_rib = sm.crossing_ideal
crossing = sm.crossing_ideal


################
# Models Dict
################


def get_models() -> dict[str, Callable[..., sax.SDict]]:
    """Return a dictionary of all models in this module."""
    models = {}
    for name, func in list(globals().items()):
        if not callable(func):
            continue
        _func = func
        while isinstance(_func, partial):
            _func = _func.func
        try:
            sig = inspect.signature(_func)
        except ValueError:
            continue
        if str(sig.return_annotation).lower().split(".")[-1] == "sdict":
            models[name] = func
    return models
