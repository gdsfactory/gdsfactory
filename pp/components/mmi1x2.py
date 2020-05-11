import pp

__version__ = "0.0.1"


@pp.autoname
def mmi1x2(
    wg_width=0.5,
    width_taper=1.0,
    length_taper=10,
    length_mmi=5.496,
    width_mmi=2.5,
    gap_mmi=0.25,
    layer=pp.LAYER.WG,
    layers_cladding=[],
    cladding_offset=3,
    with_pins=True,
):
    """ mmi 1x2

    Args:
        wg_width: input waveguides width
        width_taper: interface between input waveguides and mmi region
        length_taper: into the mmi region
        length_mmi: in x direction
        width_mmi: in y direction
        gap_mmi:  gap between tapered wg
        layer: gds layer

    .. plot::
      :include-source:

      import pp
      c = pp.c.mmi1x2(width_mmi=2, length_mmi=2.8)
      pp.plotgds(c)

    """
    c = pp.Component()
    c.wg_width = wg_width
    w_mmi = width_mmi
    w_taper = width_taper

    taper = pp.c.taper(
        length=length_taper,
        width1=wg_width,
        width2=w_taper,
        layer=layer,
        with_pins=False,
    )

    a = gap_mmi / 2 + width_taper / 2
    mmi = c << pp.c.rectangle(
        size=(length_mmi, w_mmi),
        layer=layer,
        ports_parameters={
            "E": [(w_mmi / 2 - a, w_taper), (w_mmi / 2 + a, w_taper)],
            "W": [(w_mmi / 2, w_taper)],
        },
    )
    mmi.y = 0

    for layer_cladding in layers_cladding:
        clad = c << pp.c.rectangle(
            size=(length_mmi, w_mmi + 2 * cladding_offset), layer=layer_cladding
        )
        clad.y = 0

    # For each port on the MMI rectangle
    for port_name, port in mmi.ports.items():

        # Create a taper
        taper_ref = c.add_ref(taper)

        # Connect the taper to the mmi section
        taper_ref.connect(port="2", destination=mmi.ports[port_name])

        # Add the taper port
        c.add_port(name=port_name, port=taper_ref.ports["1"])

    c.move(origin=c.ports["W0"].position, destination=(0, 0))
    if with_pins:
        pp.add_pins(c)
    return c


@pp.autoname
def mmi1x2_biased(
    wg_width=0.5,
    width_taper=1.0,
    length_taper=10,
    length_mmi=5.496,
    width_mmi=2.5,
    gap_mmi=0.25,
    layer=pp.LAYER.WG,
):
    return mmi1x2(
        wg_width=pp.bias.width(wg_width),
        width_taper=pp.bias.width(width_taper),
        length_taper=length_taper,
        length_mmi=length_mmi,
        width_mmi=pp.bias.width(width_mmi),
        gap_mmi=pp.bias.gap(gap_mmi),
        layer=layer,
    )


if __name__ == "__main__":
    c = mmi1x2()
    print(c.ports)
    # c = mmi1x2_biased()
    # pp.write_to_libary("mmi1x2", width_mmi=10, overwrite=True)
    # print(c.get_optical_ports())
    pp.show(c)
    # print(c.get_settings())
