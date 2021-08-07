"""
Routing metal lines to pads

"""

import gdsfactory

if __name__ == "__main__":
    c = gdsfactory.Component()
    ncols = 8
    nrows = 8
    pad_pitch = 150.0
    pad_width = 80
    nheaters = ncols * nrows
    heaters = c << gdsfactory.components.array(
        component=gdsfactory.c.straight_with_heater(
            port_orientation_input=180, port_orientation_output=0
        ),
        pitch=80,
        axis="y",
        n=nheaters,
    )
    pads = c << gdsfactory.components.pad_array_2d(
        ncols=ncols,
        nrows=nrows,
        pitchx=pad_pitch,
        pitchy=pad_pitch,
        port_list=("N",),
        pad_settings=dict(width=pad_width),
    )

    pads.y = 0
    heaters.y = 0
    pads.xmax = heaters.xmin - 1000

    # metal_routes = gdsfactory.routing.get_bundle(
    #     heaters.get_ports_list(port_type="dc", orientation=180),
    #     pads.get_ports_list(),
    #     waveguide="metal_routing",
    # )
    # for metal_route in metal_routes:
    #     c.add(metal_route.references)

    metal_routes, ports = gdsfactory.routing.route_ports_to_side(
        pads, side="east", waveguide="metal_routing", separation=10, width=5
    )

    for route in metal_routes:
        c.add(route.references)
    c.show()
