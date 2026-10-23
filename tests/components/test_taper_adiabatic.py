import pytest

import gdsfactory as gf


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
