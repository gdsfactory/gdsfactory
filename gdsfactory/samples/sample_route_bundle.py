if __name__ == "__main__":
    import gdsfactory as gf

    # gf.CONF.pdk = 'gdsfactory.gpdk'

    pitch = 127.0
    xs_top = [-100, -90, -80, 0, 10, 20, 40, 50, 80, 90, 100, 105, 110, 115]
    N = len(xs_top)
    xs_bottom = [(i - N / 2) * pitch for i in range(N)]
    layer = 1

    top_ports = [
        gf.Port(
            name=f"top_{i}",
            center=(xs_top[i], 0),
            width=0.5,
            orientation=270,
            layer=layer,
        )
        for i in range(N)
    ]

    bot_ports = [
        gf.Port(
            name=f"bot_{i}",
            center=(xs_bottom[i], -400),
            width=0.5,
            orientation=90,
            layer=layer,
        )
        for i in range(N)
    ]

    c = gf.Component(name="test_route_bundle")
    routes = gf.routing.route_bundle(
        c,
        top_ports,
        bot_ports,
        start_straight_length=5,
        end_straight_length=10,
        cross_section="strip",
    )
    lengths = {i: route.length for i, route in enumerate(routes)}
