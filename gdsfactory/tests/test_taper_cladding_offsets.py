from functools import partial

import gdsfactory as gf

xc_sin = partial(
    gf.cross_section.cross_section,
    width=1.0,
    layer=(0, 1),
    cladding_layers=[(0, 2), (0, 3)],
    cladding_offsets=[5, 10],
)

xc_sin_ec = partial(xc_sin, width=0.2)


@gf.cell
def demo_taper_cladding_offsets():
    taper_length = 10

    in_stub_length = 10
    out_stub_length = 10

    c = gf.Component()

    wg_in = c << gf.components.straight(length=in_stub_length, cross_section=xc_sin_ec)

    taper = c << gf.components.taper_cross_section_linear(
        length=taper_length, cross_section1=xc_sin_ec, cross_section2=xc_sin
    )

    wg_out = c << gf.components.straight(length=out_stub_length, cross_section=xc_sin)

    taper.connect("o1", wg_in.ports["o2"])
    wg_out.connect("o1", taper.ports["o2"])

    c.add_port("o1", port=wg_in.ports["o1"])
    c.add_port("o2", port=wg_out.ports["o2"])
    return c


def test_taper_cladding_offets():
    c = demo_taper_cladding_offsets()
    assert len(c.get_polygons()) == 9


if __name__ == "__main__":
    test_taper_cladding_offets()
    c = demo_taper_cladding_offsets()
    c.show()
