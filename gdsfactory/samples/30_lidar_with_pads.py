"""LiDAR demo with pads.

Exercise1. increase the number of elements of the phase array.

Exercise2. Make a PCell.

"""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component("lidar")
    elements = 2**2
    # elements = 2**4
    antenna_pitch = 2.0
    splitter_tree_spacing = (50.0, 70.0)

    splitter_tree = c << gf.components.splitter_tree(
        noutputs=elements, spacing=splitter_tree_spacing
    )
    phase_shifter = gf.components.straight_heater_meander()

    phase_shifter_optical_ports = []
    phase_shifter_electrical_ports_west = []
    phase_shifter_electrical_ports_east = []

    for i, port in enumerate(
        splitter_tree.get_ports_list(orientation=0, port_type="optical")
    ):
        ref = c.add_ref(phase_shifter, alias=f"ps{i}")
        ref.connect("o1", port)
        c.add_ports(ref.get_ports_list(port_type="electrical"), prefix=f"ps{i}")
        phase_shifter_optical_ports.append(ref.ports["o2"])
        phase_shifter_electrical_ports_west.append(ref.ports["e1"])
        phase_shifter_electrical_ports_east.append(ref.ports["e2"])

    antennas = c << gf.components.array(
        gf.components.dbr(n=200), rows=elements, columns=1, spacing=(0, antenna_pitch)
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

    pads1 = c << gf.components.array(gf.components.pad, rows=elements, columns=1)
    pads1.xmax = splitter_tree.xmin - 10
    pads1.y = 0
    ports1 = pads1.get_ports_list(orientation=0)
    routes = gf.routing.get_bundle_electrical(
        ports1=ports1, ports2=phase_shifter_electrical_ports_west, separation=20
    )
    for route in routes:
        c.add(route.references)
    c.show(show_ports=True)
