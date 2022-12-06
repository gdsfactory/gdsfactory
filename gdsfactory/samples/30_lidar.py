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
    splitter_tree = c << gf.components.splitter_tree(
        noutputs=noutputs, spacing=splitter_tree_spacing
    )

    # phase Shifters
    phase_shifter = gf.components.straight_heater_meander()
    phase_shifter_optical_ports = []

    for i, port in enumerate(
        splitter_tree.get_ports_list(orientation=0, port_type="optical")
    ):
        ref = c.add_ref(phase_shifter, alias=f"ps{i}")
        ref.connect("o1", port)
        c.add_ports(ref.get_ports_list(port_type="electrical"), prefix=f"ps{i}")
        phase_shifter_optical_ports.append(ref.ports["o2"])

    # antennas
    antennas = c << gf.components.array(
        gf.components.dbr(n=200), rows=noutputs, columns=1, spacing=(0, antenna_pitch)
    )
    antennas.xmin = ref.xmax + 50
    antennas.y = 0

    routes = gf.routing.get_bundle(
        ports1=antennas.get_ports_list(orientation=180),
        ports2=phase_shifter_optical_ports,
        radius=5,
    )

    for route in routes:
        c.add(route.references)

    c.show(show_ports=True)
