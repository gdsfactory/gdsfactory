"""Bundle routing keeps both conductors of an asymmetric cross section connected."""

import pytest

import gdsfactory as gf


@pytest.mark.parametrize("explicit_bends", [False, True])
@pytest.mark.parametrize("dy", [-100, 100])
def test_asymmetric_bend_pair(explicit_bends: bool, dy: float) -> None:
    xs = gf.cross_section.cross_section(
        width=None, sections=(((4589, 0), -1, 1), ((4589, 0), 3, 5)), radius=8
    )
    pad = gf.path.extrude(
        gf.path.straight(10), cross_section=xs, ports={0: ("o1", "o2", "optical")}
    )
    c = gf.Component()
    start, end = c << pad, c << pad
    end.dmove((100, dy))
    bend = (
        (
            gf.c.bend_euler(cross_section=xs, angle=90),
            gf.c.bend_euler(cross_section=xs, angle=-90),
        )
        if explicit_bends
        else "bend_euler"
    )
    routes = gf.routing.route_bundle(
        c,
        [start.ports["o2"]],
        [end.ports["o1"]],
        cross_section=xs,
        bend=bend,
        raise_on_error=True,
    )
    assert len(routes) == 1
    assert routes[0].n_bend90 == 2
    region = gf.kdb.Region(c.kdb_cell.begin_shapes_rec(gf.get_layer((4589, 0))))
    assert region.merged().count() == 2
    for terminal in (start, end):
        # Both bands reach each pad's outer face, on the correct side of the core.
        x, y = terminal.dxmin, terminal.dymin
        for band_y in (y + 1, y + 5):
            probe = gf.kdb.DBox(x, band_y - 0.5, x + 1, band_y + 0.5)
            assert (gf.kdb.Region(c.kcl.to_dbu(probe)) - region).is_empty()
