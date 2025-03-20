"""LiDAR demo with pads.

Exercise1. increase the number of elements of the phase array.

Exercise2. Make a PCell.

"""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component()
    elements = 2**2
    # elements = 2**4
    antenna_pitch = 2.0
    splitter_tree_spacing = (50.0, 70.0)

    splitter_tree = c << gf.components.splitter_tree(
        noutputs=elements, spacing=splitter_tree_spacing
    )
    phase_shifter = gf.components.straight_heater_meander()
    phase_shifter_extended = gf.components.extend_ports(phase_shifter, length=20)

    phase_shifter_optical_ports: list[gf.Port] = []
    phase_shifter_electrical_ports_west: list[gf.Port] = []
    phase_shifter_electrical_ports_east: list[gf.Port] = []

    for i, port in enumerate(
        splitter_tree.ports.filter(orientation=0, port_type="optical")
    ):
        ref = c.add_ref(phase_shifter_extended, name=f"ps{i}")
        ref.mirror()
        ref.connect("o1", port)
        c.add_ports(ref.ports.filter(port_type="electrical"), prefix=f"ps{i}")
        phase_shifter_optical_ports.append(ref["o2"])
        phase_shifter_electrical_ports_west.append(ref["l_e1"])
        phase_shifter_electrical_ports_east.append(ref["r_e1"])

    antennas = c << gf.components.array(
        gf.components.dbr(n=200),
        rows=elements,
        columns=1,
        column_pitch=0,
        row_pitch=antenna_pitch,
    )
    antennas.xmin = ref.xmax + 50
    antennas.mirror_y()
    antennas.y = 0

    routes = gf.routing.route_bundle(
        c,
        ports1=antennas.ports.filter(orientation=180),
        ports2=phase_shifter_optical_ports,
        radius=5,
        sort_ports=True,
        cross_section="strip",
    )

    pads1 = c << gf.components.array(
        gf.components.pad, rows=len(phase_shifter_electrical_ports_west), columns=1
    )
    pads1.xmax = splitter_tree.xmin - 10
    pads1.y = 0
    ports1 = pads1.ports.filter(orientation=0, port_type="electrical")
    routes = gf.routing.route_bundle_electrical(
        c,
        ports1=ports1,
        ports2=phase_shifter_electrical_ports_west,
        sort_ports=True,
        cross_section="metal_routing",
    )
    c.show()
