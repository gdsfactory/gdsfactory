from __future__ import annotations

import pytest

import gdsfactory as gf


def assert_endpoint_straights(
    sbend: gf.ComponentReference,
    start_straight_length: float,
    end_straight_length: float,
    bend_size: tuple[float, float],
) -> None:
    start_straight, bend, end_straight = sbend.cell.insts

    assert bend.cell.settings["size"] == bend_size
    assert start_straight.cell.settings["length"] == start_straight_length
    assert end_straight.cell.settings["length"] == end_straight_length


def test_route_bundle_sbend() -> None:
    c = gf.Component(name="test_route_bundle_sbend")
    mmi1 = c << gf.components.mmi1x2()
    mmi2 = c << gf.components.mmi1x2()
    mmi2.movex(50)
    mmi2.movey(5)

    gf.routing.route_bundle_sbend(c, mmi1.ports["o2"], mmi2.ports["o1"])
    assert len(c.insts) == 3


def test_route_single_sbend_endpoint_straights() -> None:
    c = gf.Component()
    left = c << gf.components.straight(length=10)
    right = c << gf.components.straight(length=10)
    right.movex(50)
    right.movey(5)

    sbend = gf.routing.route_single_sbend(
        c,
        left.ports["o2"],
        right.ports["o1"],
        start_straight_length=3,
        end_straight_length=7,
    )

    assert_endpoint_straights(sbend, 3, 7, (30, 5))


def test_route_single_sbend_without_endpoint_straights() -> None:
    c = gf.Component()
    left = c << gf.components.straight(length=10)
    right = c << gf.components.straight(length=10)
    right.movex(50)
    right.movey(5)

    sbend = gf.routing.route_single_sbend(
        c,
        left.ports["o2"],
        right.ports["o1"],
    )

    assert tuple(sbend.ports[0].center) == tuple(left.ports["o2"].center)
    assert tuple(sbend.ports[1].center) == tuple(right.ports["o1"].center)


def test_route_bundle_sbend_endpoint_straights() -> None:
    c = gf.Component()
    left = c << gf.components.straight(length=10)
    right = c << gf.components.straight(length=10)
    right.movex(50)
    right.movey(5)

    gf.routing.route_bundle_sbend(
        c,
        left.ports["o2"],
        right.ports["o1"],
        start_straight_length=3,
        end_straight_length=7,
    )

    assert_endpoint_straights(c.insts[-1], 3, 7, (30, 5))


def test_route_bundle_sbend_non_orthogonal() -> None:
    c = gf.Component(name="test_route_bundle_sbend_non_orthogonal")
    mmi1 = c << gf.components.mmi1x2()
    mmi2 = c << gf.components.mmi1x2()
    mmi2.rotate(45)

    with pytest.raises(ValueError, match="Ports need to have orthogonal orientation"):
        gf.routing.route_bundle_sbend(c, mmi1.ports["o2"], mmi2.ports["o1"])


def test_route_single_sbend_non_orthogonal() -> None:
    c = gf.Component(name="test_route_single_sbend_non_orthogonal")
    mmi1 = c << gf.components.mmi1x2()
    mmi2 = c << gf.components.mmi1x2()
    mmi2.rotate(45)

    with pytest.raises(ValueError, match="Ports need to have orthogonal orientation"):
        gf.routing.route_single_sbend(c, mmi1.ports["o2"], mmi2.ports["o1"])
