from typing import Callable, List, Optional, Tuple

import pp
from pp.component import Component
from pp.components.taper import taper as taper_function


@pp.cell
def mmi1x2(
    wg_width: float = 0.5,
    width_taper: float = 1.0,
    length_taper: float = 10.0,
    length_mmi: float = 5.496,
    width_mmi: float = 2.5,
    gap_mmi: float = 0.25,
    layer: Tuple[int, int] = pp.LAYER.WG,
    layers_cladding: Optional[List[Tuple]] = [pp.LAYER.WGCLAD],
    taper: Callable = taper_function,
    cladding_offset: float = 3.0,
) -> Component:
    r"""Mmi 1x2.

    Args:
        wg_width: input waveguides width
        width_taper: interface between input waveguides and mmi region
        length_taper: into the mmi region
        length_mmi: in x direction
        width_mmi: in y direction
        gap_mmi:  gap between tapered wg
        layer: gds layer
        layers_cladding: list of layers
        taper: taper function
        cladding_offset: for taper

    .. plot::
      :include-source:

      import pp
      c = pp.c.mmi1x2(width_mmi=2, length_mmi=2.8)
      pp.plotgds(c)


    .. code::

               length_mmi
                <------>
                ________
               |        |
               |         \__
               |          __
            __/          /_ _ _ _
            __          | _ _ _ _| gap_mmi
              \          \__
               |          __
               |         /
               |________|

             <->
        length_taper

    """
    c = pp.Component()
    w_mmi = width_mmi
    w_taper = width_taper

    taper = taper(
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
        centered=True,
        ports={
            "E": [(+length_mmi / 2, -a, w_taper), (+length_mmi / 2, +a, w_taper)],
            "W": [(-length_mmi / 2, 0, w_taper)],
        },
    )

    if layers_cladding:
        for layer_cladding in layers_cladding:
            clad = c << pp.c.rectangle(
                size=(length_mmi, w_mmi + 2 * cladding_offset),
                layer=layer_cladding,
                centered=True,
            )
            c.absorb(clad)

    for port_name, port in mmi.ports.items():
        taper_ref = c << taper
        taper_ref.connect(port="2", destination=mmi.ports[port_name])
        c.add_port(name=port_name, port=taper_ref.ports["1"])
        c.absorb(taper_ref)

    c.simulation_settings = dict(port_width=1.5e-6)
    c.absorb(mmi)
    return c


@pp.cell
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
    # print(c.ports)
    # print(c.get_ports_array())
    # c = mmi1x2_biased()
    # pp.write_to_libary("mmi1x2", width_mmi=10, overwrite=True)
    # print(c.get_optical_ports())
    # pp.write_gds(c, pp.CONFIG["gdsdir"] / "mmi1x2.gds")
    pp.show(c)
    # print(c.get_settings())
