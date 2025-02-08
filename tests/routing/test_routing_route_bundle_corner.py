from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory import Port


def test_connect_corner(
    data_regression: DataRegressionFixture | None, n: int = 6, config: str = "A"
) -> None:
    """Test connecting two bundles of ports in a corner.

    Args:
        data_regression: regression test data
        n: number of ports
        config: configuration of the ports
    """
    d = 10.0
    sep = 5.0
    c = gf.Component()
    layer = 1

    if config in ["A", "B"]:
        a = 100.0
        ports_A_TR = [
            Port(
                name=f"A_TR_{i}",
                center=(d, a / 2 + i * sep),
                width=0.5,
                orientation=0,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_A_TL = [
            Port(
                name=f"A_TL_{i}",
                center=(-d, a / 2 + i * sep),
                width=0.5,
                orientation=180,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_A_BR = [
            Port(
                name=f"A_BR_{i}",
                center=(d, -a / 2 - i * sep),
                width=0.5,
                orientation=0,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_A_BL = [
            Port(
                name=f"A_BL_{i}",
                center=(-d, -a / 2 - i * sep),
                width=0.5,
                orientation=180,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_A = [ports_A_TR, ports_A_TL, ports_A_BR, ports_A_BL]

        ports_B_TR = [
            Port(
                name=f"B_TR_{i}",
                center=(a / 2 + i * sep, d),
                width=0.5,
                orientation=90,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_B_TL = [
            Port(
                name=f"B_TL_{i}",
                center=(-a / 2 - i * sep, d),
                width=0.5,
                orientation=90,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_B_BR = [
            Port(
                name=f"B_BR_{i}",
                center=(a / 2 + i * sep, -d),
                width=0.5,
                orientation=270,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_B_BL = [
            Port(
                name=f"B_BL_{i}",
                center=(-a / 2 - i * sep, -d),
                width=0.5,
                orientation=270,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_B = [ports_B_TR, ports_B_TL, ports_B_BR, ports_B_BL]

    elif config in ["C", "D"]:
        a = n * sep + 2 * d
        ports_A_TR = [
            Port(
                name=f"A_TR_{i}",
                center=(a, d + i * sep),
                width=0.5,
                orientation=0,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_A_TL = [
            Port(
                name=f"A_TL_{i}",
                center=(-a, d + i * sep),
                width=0.5,
                orientation=180,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_A_BR = [
            Port(
                name=f"A_BR_{i}",
                center=(a, -d - i * sep),
                width=0.5,
                orientation=0,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_A_BL = [
            Port(
                name=f"A_BL_{i}",
                center=(-a, -d - i * sep),
                width=0.5,
                orientation=180,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_A = [ports_A_TR, ports_A_TL, ports_A_BR, ports_A_BL]

        ports_B_TR = [
            Port(
                name=f"B_TR_{i}",
                center=(d + i * sep, a),
                width=0.5,
                orientation=90,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_B_TL = [
            Port(
                name=f"B_TL_{i}",
                center=(-d - i * sep, a),
                width=0.5,
                orientation=90,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_B_BR = [
            Port(
                name=f"B_BR_{i}",
                center=(d + i * sep, -a),
                width=0.5,
                orientation=270,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_B_BL = [
            Port(
                name=f"B_BL_{i}",
                center=(-d - i * sep, -a),
                width=0.5,
                orientation=270,
                layer=layer,
            )
            for i in range(n)
        ]

        ports_B = [ports_B_TR, ports_B_TL, ports_B_BR, ports_B_BL]

    lengths = {}
    if config in ["A", "C"]:
        for ports1, ports2 in zip(ports_A, ports_B):  # type: ignore
            routes = gf.routing.route_bundle(
                c, ports1, ports2, radius=5, cross_section=gf.cross_section.strip
            )
            for i, route in enumerate(routes):
                lengths[i] = route.length

    elif config in ["B", "D"]:
        for ports1, ports2 in zip(ports_A, ports_B):  # type: ignore
            routes = gf.routing.route_bundle(
                c, ports2, ports1, radius=5, cross_section=gf.cross_section.strip
            )
            for i, route in enumerate(routes):
                lengths[i] = route.length

    if data_regression:
        data_regression.check(lengths)  # type: ignore


if __name__ == "__main__":
    test_connect_corner(None, config="A")
