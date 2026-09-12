from __future__ import annotations

import pytest

import gdsfactory as gf


@pytest.mark.parametrize("sm_width", [0.45, 0.55])
def test_mode_converter_sm_width(sm_width: float) -> None:
    component = gf.components.mode_converter(sm_width=sm_width)

    assert component.ports["o2"].width == sm_width
    assert component.ports["o4"].width == sm_width
