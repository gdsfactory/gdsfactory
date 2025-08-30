from __future__ import annotations

import pytest

import gdsfactory as gf


def test_route_single_sbend() -> None:
    c = gf.Component(name="test_route_single_sbend")
    mmi1 = c << gf.components.mmi1x2()
    mmi2 = c << gf.components.mmi1x2()
    mmi2.movex(50)
    mmi2.movey(5)

    gf.routing.route_single_sbend(c, mmi1.ports["o2"], mmi2.ports["o1"])
    assert len(c.insts) == 3


def test_route_single_sbend_non_orthogonal() -> None:
    c = gf.Component(name="test_route_single_sbend_non_orthogonal")
    mmi1 = c << gf.components.mmi1x2()
    mmi2 = c << gf.components.mmi1x2()
    mmi2.rotate(45)

    with pytest.raises(ValueError, match="Ports need to have orthogonal orientation"):
        gf.routing.route_single_sbend(c, mmi1.ports["o2"], mmi2.ports["o1"])
