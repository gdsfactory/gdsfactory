from functools import partial

import gdsfactory as gf
from gdsfactory.generic_tech.layer_map import LAYER
from gdsfactory.typings import Layer

extend_ports = gf.components.extend_ports


def test_extend_ports() -> None:
    import gdsfactory.components as pc

    width = 0.5
    xs_strip = partial(
        gf.cross_section.strip,
        width=width,
        cladding_layers=None,
    )

    c = pc.straight(cross_section=xs_strip)

    c1 = extend_ports(
        component=c,
        cross_section=xs_strip,
    )
    assert len(c.ports) == len(c1.ports)
    p = assert_polygons(c1, 3)
    c2 = extend_ports(component=c, cross_section=xs_strip, port_names=("o1",))
    p = assert_polygons(c2, 2)
    c3 = extend_ports(component=c, cross_section=xs_strip)
    p = len(c3.get_polygons()[LAYER.WG])
    assert p == 3, p
    c4 = extend_ports(component=c, port_type="electrical")
    p = assert_polygons(c4, 1)


def assert_polygons(
    component: gf.Component, n_polygons: int, layer: Layer = LAYER.WG
) -> int:
    result = len(component.get_polygons()[layer])
    assert result == n_polygons, result
    assert len(component.insts) == n_polygons, len(component.insts)
    return result


if __name__ == "__main__":
    test_extend_ports()
