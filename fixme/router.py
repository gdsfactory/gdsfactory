"""
This router could save more routing space with a more direct algorithm

"""

import gdsfactory as gf
from gdsfactory.components.extend_ports_list import extend_ports_list
from gdsfactory.components.via_stack import via_stack_heater


if __name__ == "__main__":
    ncols = 8
    nrows = 16
    N = ncols * nrows
    with_pads = True
    pad_pitch = 150.0 * 2
    metal_width = 5.0
    metal_spacing = 10.0
    cross_section = gf.cross_section.metal3
    length = 200
    dy = 100

    c = gf.Component()
    ps = gf.components.straight_heater_metal()
    ps_array = gf.components.array(component=ps, spacing=(0, 20), columns=1, rows=2)

    splitter = gf.components.splitter_tree(noutputs=N, spacing=(90, dy))
    splitters = c.add_ref(splitter)
    splitters.movey(-30)
    splitters.xmax = 0

    extension_factory = gf.partial(
        gf.components.straight_heater_metal,
        length=length,
        port_orientation1=180,
        port_orientation2=0,
        via_stack=via_stack_heater,
    )

    ps = c << extend_ports_list(
        ports=splitters.get_ports_list(orientation=0),
        extension_factory=extension_factory,
    )

    if with_pads:
        pads = c << gf.components.array_with_fanout_2d(
            columns=ncols * 2,
            rows=nrows,
            pitch=pad_pitch,
            width=metal_width,
            waveguide_pitch=metal_spacing,
            cross_section=cross_section,
        )
        pads.rotate(180)
        pads.y = 15

        # pads.xmax = ps.xmin - 2500 # works
        pads.xmax = ps.xmin - 1500  # does not work

        routes_bend180 = gf.routing.get_routes_bend180(
            ports=ps.get_ports_list(port_type="electrical", orientation=0),
            radius=dy / 8,
            layer=(31, 0),
            width=metal_width,
        )
        c.add(routes_bend180.references)

        metal_routes = gf.routing.get_bundle(
            ps.get_ports_list(port_type="electrical", orientation=180)
            + list(routes_bend180.ports.values()),
            pads.get_ports_list(),
            width=metal_width,
            separation=metal_spacing,
            cross_section=cross_section,
        )
        for metal_route in metal_routes:
            c.add(metal_route.references)

    c.show()
