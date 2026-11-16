import numpy as np
import pytest
from scipy.integrate import quad

import gdsfactory as gf
from gdsfactory.component_functions.tapers.taper_adiabatic import (
    neff_TE1550SOI_220nm,
)
from gdsfactory.path import transition_adiabatic


@pytest.mark.parametrize(("width1", "width2"), [(0.5, 5.0), (5.0, 0.5), (0.5, 2.0)])
def test_taper_adiabatic_ends_at_port_widths(width1: float, width2: float) -> None:
    """The wide end of the polygon is as wide as its port."""
    c = gf.components.taper_adiabatic(width1=width1, width2=width2)

    assert c.ysize == pytest.approx(max(width1, width2))
    assert c.xmax == pytest.approx(c.ports["o2"].center[0])


def test_transition_adiabatic_narrowing_mirrors_widening() -> None:
    x, w = transition_adiabatic(0.5, 5.0, neff_w=neff_TE1550SOI_220nm)
    x_rev, w_rev = transition_adiabatic(5.0, 0.5, neff_w=neff_TE1550SOI_220nm)

    assert (w[0], w[-1]) == (0.5, 5.0)
    np.testing.assert_allclose(x_rev, x[-1] - x[::-1])
    np.testing.assert_allclose(w_rev, w[::-1])


def test_transition_adiabatic_length() -> None:
    """The length matches integrating dx/dw = neff(w) * w / wavelength."""
    x, _ = transition_adiabatic(0.5, 5.0, neff_w=neff_TE1550SOI_220nm)
    length, _ = quad(lambda w: neff_TE1550SOI_220nm(w) * w / 1.55, 0.5, 5.0)

    assert x[-1] == pytest.approx(length, abs=1e-3)


def test_transition_adiabatic_warns_if_max_length_too_short() -> None:
    with pytest.warns(UserWarning, match="max_length=10 is too short"):
        x, w = transition_adiabatic(
            0.5, 5.0, neff_w=neff_TE1550SOI_220nm, max_length=10
        )

    assert x[-1] == 10
    assert w[-1] < 5.0


@pytest.mark.parametrize("length", [10.0, 40.0])
@pytest.mark.parametrize(("width1", "width2"), [(0.5, 5.0), (5.0, 0.5)])
def test_taper_adiabatic_explicit_length(
    width1: float, width2: float, length: float
) -> None:
    """An explicit length stretches or compresses the optimal shape."""
    optimal = gf.components.taper_adiabatic(width1=width1, width2=width2)
    optimal_length = optimal.ports["o2"].center[0]
    c = gf.components.taper_adiabatic(width1=width1, width2=width2, length=length)

    assert c.ports["o1"].center == (0, 0)
    assert c.ports["o2"].center == (length, 0)
    assert (c.xmin, c.xmax) == pytest.approx((0, length))
    assert c.ysize == pytest.approx(optimal.ysize)
    assert c.area("WG") == pytest.approx(
        optimal.area("WG") * length / optimal_length, rel=1e-3
    )
