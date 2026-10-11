from __future__ import annotations

import pytest

import gdsfactory as gf


@pytest.mark.parametrize(
    "steps",
    [None, [{"x": 50}, {"y": 50}]],
    ids=["manhattan", "steps"],
)
def test_route_single_route_width(steps: list[dict[str, float]] | None) -> None:
    """route_width sets the width of the straights too, not only of the bends."""
    route_width = 2.0
    c = gf.Component()
    s1 = c << gf.components.straight(cross_section="strip")
    s2 = c << gf.components.straight(cross_section="strip")
    s2.move((100, 50))

    route = gf.routing.route_single(
        c,
        s1.ports["o2"],
        s2.ports["o1"],
        cross_section="strip",
        route_width=route_width,
        steps=steps,
    )

    widths = {port.width for inst in route.instances for port in inst.ports}
    assert widths == {c.kcl.to_dbu(route_width)}
