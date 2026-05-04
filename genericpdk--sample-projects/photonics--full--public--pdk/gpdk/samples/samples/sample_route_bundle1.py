import gdsfactory as gf


@gf.cell
def sample_route_bundle1() -> gf.Component:
    ys_right = [
        0,
        10,
        20,
        40,
        50,
        80,
    ]  # This line creates a list of six explicit y-coordinates for the right-side ports.
    pitch = 127.0  # Defines the constant vertical distance for the left side ports.
    N = len(
        ys_right
    )  # This defines the total number of ports in each column (N, which is 6).

    # This line uses a list comprehension to calculate six y-coordinates for the left-side ports.
    # The formula (i - N / 2) * pitch ensures that the ports are evenly spaced by 127.0 µm and are centered vertically around y=0.
    ys_left = [(i - N / 2) * pitch for i in range(N)]
    layer = (1, 0)

    right_ports = [
        gf.Port(
            f"R_{i}",
            center=(0, ys_right[i]),
            width=0.5,
            orientation=180,
            layer=gf.get_layer(layer),
        )
        for i in range(N)
    ]
    left_ports = [
        gf.Port(
            f"L_{i}",
            center=(-200, ys_left[i]),
            width=0.5,
            orientation=0,
            layer=gf.get_layer(layer),
        )
        for i in range(N)
    ]

    # You can also mess up the port order and it will sort them by default.
    left_ports.reverse()

    c = gf.Component()
    routes = gf.routing.route_bundle(
        c,
        left_ports,
        right_ports,
        start_straight_length=50,
        sort_ports=True,
        cross_section="strip",
    )
    for route in routes:
        print(route.length)
    c.add_ports(left_ports)
    c.add_ports(right_ports)
    return c
