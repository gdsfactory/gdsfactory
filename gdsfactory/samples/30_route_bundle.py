"""LiDAR demo.

Exercise1. increase the number of elements of the phase array.

Exercise2. Make a PCell.

"""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component("lidar")
    noutputs = 2**2
    antenna_pitch = 2.0
    splitter_tree_spacing = (50.0, 70.0)

    # power Splitter
    ref = c << gf.components.array(
        gf.components.straight(),
        rows=noutputs,
        columns=1,
        spacing=(0, antenna_pitch * 2),
        centered=True,
    )

    # antennas
    antennas = c << gf.components.array(
        gf.components.dbr(n=200),
        rows=noutputs,
        columns=1,
        spacing=(0, antenna_pitch),
        centered=True,
    )
    antennas.d.xmin = ref.d.xmax + 100
    antennas.d.y = 0

    routes = gf.routing.route_bundle(
        c,
        ports1=antennas.ports.filter(orientation=180),
        ports2=ref.ports.filter(orientation=0),
        radius=5,
        enforce_port_ordering=False,
    )

    c.show()
