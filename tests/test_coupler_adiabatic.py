import pytest

import gdsfactory as gf


@pytest.mark.parametrize("length2", [35.0, 50.0])
def test_coupler_adiabatic_length(length2: float) -> None:
    c = gf.components.coupler_adiabatic(length2=length2)
    assert c.ports["o3"].center[0] - c.ports["o1"].center[0] == pytest.approx(
        20.0 + length2 + 30.0
    )


@pytest.mark.parametrize("wg_sep", [0.6, 1.0])
def test_coupler_adiabatic_edge_separations(wg_sep: float) -> None:
    c = gf.components.coupler_adiabatic(
        wg_sep=wg_sep, input_wg_sep=4.0, output_wg_sep=5.0
    )
    for bottom, top, gap in [("o1", "o2", 4.0), ("o4", "o3", 5.0)]:
        p1, p2 = c.ports[bottom], c.ports[top]
        assert abs(p2.center[1] - p1.center[1]) - (
            p1.width + p2.width
        ) / 2 == pytest.approx(gap, abs=0.002)
