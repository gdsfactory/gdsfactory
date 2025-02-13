"""LiDAR demo.

Exercise1. increase the number of noutputs of the phase array.

Exercise2. Make a PCell.

"""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component()
    noutputs = 2**2
    antenna_pitch = 2.0
    splitter_tree_spacing = (50.0, 70.0)

    # power Splitter
    ref = c << gf.components.array(
        gf.components.straight(),
        rows=noutputs,
        columns=1,
        column_pitch=0,
        row_pitch=40,
        centered=True,
    )

    # antennas
    antennas = c << gf.components.array(
        gf.components.dbr(n=200),
        rows=noutputs,
        columns=1,
        column_pitch=0,
        row_pitch=antenna_pitch,
        centered=True,
    )
    antennas.dxmin = ref.dxmax + 100
    antennas.dy = 0

    routes = gf.routing.route_bundle(
        c,
        ports1=antennas.ports.filter(orientation=180),
        ports2=ref.ports.filter(orientation=0),
        radius=5,
        sort_ports=True,
        cross_section="strip",
    )

    c.show()
