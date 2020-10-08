import pp
from pp.component import Component
from typing import Any, List, Tuple


@pp.autoname
def mmi1x2(
    wg_width: float = 0.5,
    width_taper: float = 1.0,
    length_taper: float = 10.0,
    length_mmi: float = 5.496,
    width_mmi: float = 2.5,
    gap_mmi: float = 0.25,
    layer: Tuple[int, int] = pp.LAYER.WG,
    layers_cladding: List[Any] = [],
    cladding_offset: float = 3.0,
) -> Component:
    """mmi 1x2

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
        layers_cladding=layers_cladding,
        cladding_offset=cladding_offset,
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
        c.absorb(clad)

    # For each port on the MMI rectangle
    for port_name, port in mmi.ports.items():

        # Create a taper
        taper_ref = c.add_ref(taper)

        # Connect the taper to the mmi section
        taper_ref.connect(port="2", destination=mmi.ports[port_name])

        # Add the taper port
        c.add_port(name=port_name, port=taper_ref.ports["1"])
        c.absorb(taper_ref)

    c.move(origin=c.ports["W0"].position, destination=(0, 0))
    c.simulation_settings = dict(port_width=1.5e-6)
    c.absorb(mmi)

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
    c = mmi1x2(pins=True)
    print(c.ports)
    # print(c.get_ports_array())
    # c = mmi1x2_biased()
    # pp.write_to_libary("mmi1x2", width_mmi=10, overwrite=True)
    # print(c.get_optical_ports())
    pp.write_gds(c, pp.CONFIG["gdsdir"] / "mmi1x2.gds")
    pp.show(c)
    # print(c.get_settings())
