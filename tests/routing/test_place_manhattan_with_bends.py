from __future__ import annotations

import kfactory as kf

import gdsfactory as gf
from gdsfactory.routing.place_manhattan_with_bends import place_manhattan_with_bends


def test_place_manhattan_with_bends_uses_bends_in_order() -> None:
    c = gf.Component()
    bends = [
        gf.components.bend_circular(radius=5),
        gf.components.bend_euler(radius=10),
        gf.components.bend_euler(radius=20),
    ]
    source = c << gf.components.straight(length=10)
    target = c << gf.components.straight(length=10)
    source.dmovex(-10)
    target.dmove((400, 200))
    p1 = source.ports["o2"]
    p2 = target.ports["o1"]

    def straight_factory(width: int, length: int) -> gf.Component:
        return gf.components.straight(
            width=c.kcl.to_um(width), length=c.kcl.to_um(length)
        )

    route = place_manhattan_with_bends(
        c.to_itype(),
        p1.to_itype(),
        p2.to_itype(),
        pts=[
            kf.kdb.Point(0, 0),
            kf.kdb.Point(100_000, 0),
            kf.kdb.Point(100_000, 200_000),
            kf.kdb.Point(200_000, 200_000),
            kf.kdb.Point(200_000, 100_000),
            kf.kdb.Point(300_000, 100_000),
            kf.kdb.Point(300_000, 200_000),
            kf.kdb.Point(400_000, 200_000),
        ],
        bend90_cells=[bend.to_itype() for bend in bends],
        straight_factory=straight_factory,
        route_width=500,
    )

    bend_names = [
        inst.cell.name for inst in route.instances if inst.cell.name.startswith("bend_")
    ]
    assert len(bend_names) == 6
    assert bend_names[0] == bends[0].name
    assert bend_names[1] == bends[1].name
    assert bend_names[2:] == [bends[2].name] * 4
    assert route.n_bend90 == 6
    assert route.length > 0
