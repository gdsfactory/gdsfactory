import gdsfactory as gf
from gdsfactory.add_padding import add_padding
from gdsfactory.component import Component
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.cross_section import get_cross_section
from gdsfactory.types import ComponentFactory


@gf.cell
def mmi1x2(
    width: float = 0.5,
    width_taper: float = 1.0,
    length_taper: float = 10.0,
    length_mmi: float = 5.5,
    width_mmi: float = 2.5,
    gap_mmi: float = 0.25,
    taper: ComponentFactory = taper_function,
    with_cladding_box: bool = True,
    waveguide: str = "strip",
    **kwargs
) -> Component:
    r"""Mmi 1x2.

    Args:
        width: input and output straight width
        width_taper: interface between input straights and mmi region
        length_taper: into the mmi region
        length_mmi: in x direction
        width_mmi: in y direction
        gap_mmi:  gap between tapered wg
        taper: taper function
        with_cladding_box: to avoid DRC acute angle errors in cladding
        kwargs: waveguide_settings


    .. code::

               length_mmi
                <------>
                ________
               |        |
               |         \__
               |          __  E1
            __/          /_ _ _ _
         W0 __          | _ _ _ _| gap_mmi
              \          \__
               |          __  E0
               |         /
               |________|

             <->
        length_taper

    """
    x = get_cross_section(waveguide, **kwargs)
    cladding_offset = x.info["cladding_offset"]
    layers_cladding = x.info["layers_cladding"]
    layer = x.info["layer"]

    c = Component()
    w_mmi = width_mmi
    w_taper = width_taper

    taper = taper(
        length=length_taper, width1=width, width2=w_taper, waveguide=waveguide, **kwargs
    )

    a = gap_mmi / 2 + width_taper / 2
    mmi = c << gf.components.rectangle(
        size=(length_mmi, w_mmi),
        layer=layer,
        centered=True,
        ports={
            "E": [(+length_mmi / 2, -a, w_taper), (+length_mmi / 2, +a, w_taper)],
            "W": [(-length_mmi / 2, 0, w_taper)],
        },
    )

    for port_name, port in mmi.ports.items():
        taper_ref = c << taper
        taper_ref.connect(port="2", destination=port)
        c.add_port(name=port_name, port=taper_ref.ports["1"])
        c.absorb(taper_ref)

    c.simulation_settings = dict(port_width=1.5e-6)
    c.absorb(mmi)

    layers_cladding = layers_cladding or []
    if layers_cladding and with_cladding_box:
        add_padding(
            c,
            default=cladding_offset,
            right=0,
            left=0,
            top=cladding_offset,
            bottom=cladding_offset,
            layers=layers_cladding,
        )
    return c


if __name__ == "__main__":

    c = mmi1x2(waveguide="nitride")
    c.show()

    # print(c.ports)
    # c = mmi1x2_biased()
    # print(c.get_optical_ports())
    # c.write_gds(gf.CONFIG["gdsdir"] / "mmi1x2.gds")
