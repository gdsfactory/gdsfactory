"""LiDAR demo.

Exercise1. increase the number of elements of the phase array.

Exercise2. Make a PCell.

"""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component(name="lidar")
    noutputs = 2**2
    antenna_pitch = 2.0
    splitter_tree_spacing = (50.0, 70.0)

    # power Splitter
    splitter_tree = c << gf.components.splitter_tree(
        noutputs=noutputs, spacing=splitter_tree_spacing
    )

    # phase Shifters
    phase_shifter = gf.components.straight_heater_meander()
    phase_shifter_extended = gf.components.extend_ports(phase_shifter, length=20)
    phase_shifter_optical_ports: list[gf.Port] = []

    for i, port in enumerate(
        splitter_tree.ports.filter(orientation=0, port_type="optical")
    ):
        ref = c.add_ref(phase_shifter_extended, name=f"ps{i}")
        ref.mirror()
        ref.connect("o1", port)
        c.add_ports(ref.ports.filter(port_type="electrical"), prefix=f"ps{i}")
        phase_shifter_optical_ports.append(ref.ports["o2"])

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
        ports1=list(antennas.ports.filter(orientation=180)),
        ports2=phase_shifter_optical_ports,
        radius=5,
        sort_ports=True,
        cross_section="strip",
    )

    c.show()
