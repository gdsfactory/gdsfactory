"""Some manhattan routes have disconnected waveguides."""

import gdsfactory as gf


if __name__ == "__main__":
    c = gf.Component()
    c1 = c << gf.components.array(rows=2, columns=1, spacing=(0, 100))
    c2 = c << gf.components.array(rows=2, columns=1, spacing=(0, 5))

    c2.movex(200)
    c1.y = 0
    c2.y = 0

    routes = gf.routing.get_bundle_path_length_match(
        c1.get_ports_list(orientation=0),
        c2.get_ports_list(orientation=180),
        end_straight_length=0,
        start_straight_length=0,
        separation=50,
        # modify_segment_i=-3,
        radius=10
        # radius=5  # smaller radius works
    )

    for route in routes:
        c.add(route.references)

    c.show(show_ports=True)
