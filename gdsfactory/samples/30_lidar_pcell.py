"""LiDAR demo.

Exercise1. increase the number of elements of the phase array.

Exercise2. Make a PCell.

"""

from __future__ import annotations

import gdsfactory as gf


@gf.cell
def lidar(
    noutputs=2**2, antenna_pitch=2.0, splitter_tree_spacing=(50.0, 70.0)
) -> gf.Component:
    """LiDAR demo.

    Args:
        noutputs: number of outputs.
        antenna_pitch: pitch of the antennas.
        splitter_tree_spacing: spacing of the splitter tree.
    """
    c = gf.Component("lidar")

    # power Splitter
    splitter_tree = c << gf.components.splitter_tree(
        noutputs=noutputs, spacing=splitter_tree_spacing
    )

    # phase Shifters
    phase_shifter = gf.components.straight_heater_meander()
    phase_shifter_optical_ports = []

    for i, port in enumerate(
        splitter_tree.ports.filter(orientation=0, port_type="optical")
    ):
        ref = c.add_ref(phase_shifter, name=f"ps{i}")
        ref.connect("o1", port)
        c.add_ports(ref.ports.filter(port_type="electrical"), prefix=f"ps{i}")
        phase_shifter_optical_ports.append(ref.ports["o2"])

    # antennas
    antennas = c << gf.components.array(
        gf.components.dbr(n=200), rows=noutputs, columns=1, spacing=(0, antenna_pitch)
    )
    antennas.dxmin = ref.dxmax + 50
    antennas.dy = 10

    gf.routing.route_bundle(
        c,
        ports1=antennas.ports.filter(orientation=180),
        ports2=phase_shifter_optical_ports,
        radius=5,
        sort_ports=True,
    )

    return c


if __name__ == "__main__":
    c = lidar(noutputs=2**4)
    c.show()
