"""
This router could save more routing space leveraging routing with different metal layers

"""

import gdsfactory as gf
from gdsfactory.components.extend_ports_list import extend_ports_list
from gdsfactory.components.via_stack import via_stack_heater0, via_stack_heater180


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

    c = gf.Component()
    ps = gf.components.straight_heater_metal()
    ps_array = gf.components.array(component=ps, pitch=20)
    dy = 100

    splitter = gf.components.splitter_tree(noutputs=N, dx=80, dy=dy)
    splitters = c.add_ref(splitter)
    splitters.movey(-30)
    splitters.xmax = 0

    extension_factory = gf.partial(
        gf.components.straight_heater_metal,
        length=length,
        via_stack1=via_stack_heater180,
        via_stack2=via_stack_heater0,
    )

    ps = c << extend_ports_list(
        ports=splitters.get_ports_list(orientation=0),
        extension_factory=extension_factory,
    )

    if with_pads:
        pads = c << gf.components.array_with_fanout_2d(
            cols=ncols * 2,
            rows=nrows,
            pitch=pad_pitch,
            width=metal_width,
            waveguide_pitch=metal_spacing,
        )
        pads.rotate(180)
        pads.y = 15

        pads.xmax = ps.xmin - 2500

        routes_bend180 = gf.routing.get_routes_bend180(
            ports=ps.get_ports_list(port_type="electrical", orientation=0),
            radius=dy / 8,
            width=metal_width,
            layer=(31, 0),
        )
        c.add(routes_bend180.references)

        metal_routes = gf.routing.get_bundle(
            ps.get_ports_list(port_type="electrical", orientation=180)
            + list(routes_bend180.ports.values()),
            pads.get_ports_list(),
            width=metal_width,
            separation=metal_spacing,
            layer=(31, 0),
        )
        for metal_route in metal_routes:
            c.add(metal_route.references)

    c.show()
