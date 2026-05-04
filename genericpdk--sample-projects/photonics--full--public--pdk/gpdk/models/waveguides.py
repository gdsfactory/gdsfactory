"""SAX models for Sparameter circuit simulations."""

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


def straight_strip(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
) -> sax.SDict:
    """Straight strip waveguide model."""
    return sm.straight(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        wl0=1.55,
        neff=2.38,
        ng=4.30,
    )


def straight_rib(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 3.0,
) -> sax.SDict:
    """Straight rib waveguide model."""
    return sm.straight(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
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
    **kwargs,
) -> sax.SDict:
    """Euler bend model."""
    # NOTE: it is assumed that `bend_euler` exposes it's length in its info dictionary!
    return straight(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


def bend_euler_strip(
    *, wl: Float = 1.55, length: float = 10.0, loss_dB_cm: float = 3, **kwargs
) -> sax.SDict:
    """Euler bend strip model."""
    return bend_euler(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section="strip",
    )


def bend_euler_rib(
    *, wl: Float = 1.55, length: float = 10.0, loss_dB_cm: float = 3, **kwargs
) -> sax.SDict:
    """Euler bend rib model."""
    return bend_euler(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section="rib",
    )


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


def taper_rib(
    *,
    wl: Float = 1.55,
    length: float = 10.0,
    loss_dB_cm: float = 0.0,
) -> sax.SDict:
    """Taper rib model."""
    return taper(
        wl=wl,
        length=length,
        loss_dB_cm=loss_dB_cm,
        cross_section="rib",
    )


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


def trans_rib10(
    *,
    wl: Float = 1.55,
    loss_dB_cm: float = 0.0,
    cross_section="strip",
) -> sax.SDict:
    """Taper strip to ridge 10um model."""
    return taper_strip_to_ridge(
        wl=wl,
        length=10.0,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


def trans_rib20(
    *,
    wl: Float = 1.55,
    loss_dB_cm: float = 0.0,
    cross_section="strip",
) -> sax.SDict:
    """Taper strip to ridge 20um model."""
    return taper_strip_to_ridge(
        wl=wl,
        length=20.0,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )


def trans_rib50(
    *,
    wl: Float = 1.55,
    loss_dB_cm: float = 0.0,
    cross_section="strip",
) -> sax.SDict:
    """Taper strip to ridge 50um model."""
    return taper_strip_to_ridge(
        wl=wl,
        length=50.0,
        loss_dB_cm=loss_dB_cm,
        cross_section=cross_section,
    )
