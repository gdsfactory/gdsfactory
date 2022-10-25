"""FIXME. Some manhattan routes have disconnected waveguides.

Now we raise a warning.

ideally we also enable 180 deg routing.

"""

import gdsfactory as gf


if __name__ == "__main__":
    c = gf.Component()
    # c1 = c << gf.components.straight_array(spacing=200)
    c1 = c << gf.components.array(spacing=(0, 100), rows=2, columns=1)
    c2 = c << gf.components.array(spacing=(0, 5), rows=2, columns=1)

    c2.movex(200)
    c1.y = 0

    routes = gf.routing.get_bundle_path_length_match(
        c1.get_ports_list(orientation=0),
        c2.get_ports_list(orientation=180),
        end_straight_length=0,
        start_straight_length=0,
        separation=50,
    )

    for route in routes:
        c.add(route.references)

    c.show(show_ports=True)
