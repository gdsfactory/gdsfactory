from functools import partial

import gdsfactory as gf

layer_bbox = (1, 0)


@gf.cell
def sample_add_pads() -> gf.Component:
    """Sample component."""
    c = gf.Component()
    c.add_polygon(
        [[0.0, 25.0], [374.0, 25.0], [374.0, -25.0], [0.0, -25.0]], layer=layer_bbox
    )

    c.add_port(
        name="g1",
        width=100.0,
        cross_section="xs_m3",
        center=(57.0, 25.0),
        orientation=90,
        port_type="electrical",
    )
    c.add_port(
        name="g2",
        width=100.0,
        cross_section="xs_m3",
        center=(57.0, -25.0),
        orientation=-90,
        port_type="electrical",
    )
    c.add_port(
        name="g3",
        width=100.0,
        cross_section="xs_m3",
        center=(317.0, 25.0),
        orientation=90,
        port_type="electrical",
    )
    c.add_port(
        name="g4",
        width=100.0,
        cross_section="xs_m3",
        center=(317.0, -25.0),
        orientation=-90,
        port_type="electrical",
    )
    c.add_port(
        name="o1",
        width=2.0,
        cross_section="xs_sc",
        center=(0.0, 0.0),
        orientation=180,
        port_type="optical",
    )
    c.add_port(
        name="o2",
        width=2.0,
        cross_section="xs_sc",
        center=(374.0, 0.0),
        orientation=0,
        port_type="optical",
    )
    c.add_port(
        name="s1",
        width=60.0,
        cross_section="xs_m3",
        center=(187.0, 25.0),
        orientation=90,
        port_type="electrical",
    )
    c.add_port(
        name="s2",
        width=60.0,
        cross_section="xs_m3",
        center=(187.0, -25.0),
        orientation=-90,
        port_type="electrical",
    )
    return c


def test_sample_add_pads() -> None:
    p = partial(gf.c.pad, size=(100, 100))
    c = sample_add_pads()
    cc = gf.routing.add_pads_top(
        c,
        straight_separation=26,
        pad_spacing=150,
        pad=p,
    )
    assert cc


if __name__ == "__main__":
    p = partial(gf.components.pad, size=(100, 100))
    c = sample_add_pads()
    cc = gf.routing.add_pads_top(
        c,
        straight_separation=26,
        pad_spacing=150,
        pad=p,
    )
    cc.show()
