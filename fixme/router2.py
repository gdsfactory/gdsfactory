"""
This router could save more routing space leveraging routing with different metal layers

"""

import pp
from pp.components.extend_ports_list import extend_ports_list


if __name__ == "__main__":
    ncols = 8
    nrows = 16
    N = ncols * nrows
    with_pads = False
    with_pads = True
    pad_pitch = 150.0
    metal_width = 5.0
    metal_spacing = 10.0
    length = 200

    c = pp.Component()
    ps = pp.c.straight_with_heater()
    ps_array = pp.c.array(component=ps, pitch=20)
    dy = 100

    splitter = pp.c.splitter_tree(noutputs=N, waveguide="nitride", dx=80, dy=dy)
    splitters = c.add_ref(splitter)
    splitters.movey(-30)
    splitters.xmax = 0

    ps = c << extend_ports_list(
        ports=splitters.get_ports_list(prefix="E"),
        extension_factory=pp.c.straight_with_heater,
        extension_settings=dict(
            length=length, port_orientation_input=180, port_orientation_output=0
        ),
    )

    if with_pads:
        pads = c << pp.components.array_with_fanout_2d(
            cols=ncols * 2,
            rows=nrows,
            pitch=pad_pitch,
            width=metal_width,
            waveguide_pitch=metal_spacing,
        )
        pads.rotate(180)
        pads.y = 15

        pads.xmax = ps.xmin - 2500

        routes_bend180 = pp.routing.get_routes_bend180(
            ports=ps.get_ports_list(port_type="dc", orientation=0),
            radius=dy / 8,
            waveguide="metal_routing",
            width=metal_width,
        )
        c.add(routes_bend180.references)

        metal_routes = pp.routing.get_bundle(
            ps.get_ports_list(port_type="dc", orientation=180)
            + list(routes_bend180.ports.values()),
            pads.get_ports_list(),
            waveguide="metal_routing",
            width=metal_width,
            separation=metal_spacing,
        )
        for metal_route in metal_routes:
            c.add(metal_route.references)

    c.show()
