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
    width = 5
    nheaters = ncols * nrows
    heaters = c << gdsfactory.components.array(
        component=gdsfactory.components.straight_with_heater(
            port_orientation_input=180, port_orientation_output=0
        ),
        pitch=80,
        axis="y",
        n=nheaters,
    )
    heaters.y = 0

    port_pads = []

    for row in range(nrows):
        pads = c << gdsfactory.components.array_with_fanout(
            n=ncols, pitch=pad_pitch, width=width, waveguide_pitch=width * 2
        )
        pads.rotate(180)
        pads.y = row * pad_pitch - pad_pitch * nrows / 2
        pads.xmax = heaters.xmin - 600
        port_pads.extend(pads.get_ports_list())

    with_routes = False
    with_routes = True

    if with_routes:
        metal_routes = gdsfactory.routing.get_bundle(
            heaters.get_ports_list(port_type="dc", orientation=180),
            port_pads,
            waveguide="metal_routing",
            width=width,
        )

        for route in metal_routes:
            c.add(route.references)
    c.show()
