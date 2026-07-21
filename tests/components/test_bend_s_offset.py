import pytest

import gdsfactory as gf


@pytest.mark.parametrize("offset", [40.0, -40.0])
def test_bend_s_offset_supports_signed_offsets(offset: float) -> None:
    component = gf.components.bend_s_offset(offset=offset, radius=10)
    ports = list(component.ports)

    assert len(component.get_polygons()) == 1
    assert ports[0].dcenter == (0.0, 0.0)
    assert ports[1].dcenter[0] > 0
    assert ports[1].dcenter[1] == pytest.approx(offset)
    assert ports[0].orientation == 180
    assert ports[1].orientation == 0
