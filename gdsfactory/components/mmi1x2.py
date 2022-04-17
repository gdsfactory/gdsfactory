import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.cross_section import strip
from gdsfactory.types import ComponentSpec, CrossSectionSpec


@gf.cell
def mmi1x2(
    width: float = 0.5,
    width_taper: float = 1.0,
    length_taper: float = 10.0,
    length_mmi: float = 5.5,
    width_mmi: float = 2.5,
    gap_mmi: float = 0.25,
    taper: ComponentSpec = taper_function,
    cross_section: CrossSectionSpec = strip,
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
        cross_section: specification (CrossSection, string, CrossSectionFactory, dict).


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
    gf.snap.assert_on_2nm_grid(gap_mmi)
    c = Component()
    w_mmi = width_mmi
    w_taper = width_taper

    taper = gf.get_component(
        taper,
        length=length_taper,
        width1=width,
        width2=w_taper,
        cross_section=cross_section,
    )

    a = gap_mmi / 2 + width_taper / 2
    mmi = c << gf.components.straight(
        length=length_mmi, width=w_mmi, cross_section=cross_section
    )

    ports = [
        gf.Port("o1", orientation=180, midpoint=(0, 0), width=w_taper),
        gf.Port("o2", orientation=0, midpoint=(+length_mmi, +a), width=w_taper),
        gf.Port("o3", orientation=0, midpoint=(+length_mmi, -a), width=w_taper),
    ]

    for port in ports:
        taper_ref = c << taper
        taper_ref.connect(port="o2", destination=port)
        c.add_port(name=port.name, port=taper_ref.ports["o1"])
        c.absorb(taper_ref)

    c.absorb(mmi)
    return c


if __name__ == "__main__":
    c = mmi1x2(cross_section=dict(cross_section="rib"))
    c.show()

    # c.pprint_ports()

    # c2 = gf.components.extend_ports(c)
    # c2.show()

    # print(c.ports)
    # c = mmi1x2_biased()
    # print(c.get_optical_ports())
    # c.write_gds(gf.CONFIG["gdsdir"] / "mmi1x2.gds")
    # print(c.ports["o1"].cross_section.info)
