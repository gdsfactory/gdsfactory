import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.cross_section import strip
from gdsfactory.types import ComponentFactory, CrossSectionSpec


@gf.cell
def mmi2x2(
    width: float = 0.5,
    width_taper: float = 1.0,
    length_taper: float = 10.0,
    length_mmi: float = 5.5,
    width_mmi: float = 2.5,
    gap_mmi: float = 0.25,
    taper: ComponentFactory = taper_function,
    cross_section: CrossSectionSpec = strip,
) -> Component:
    r"""Mmi 2x2.

    Args:
        width: input and output straight width
        width_taper: interface between input straights and mmi region
        length_taper: into the mmi region
        length_mmi: in x direction
        width_mmi: in y direction
        gap_mmi: (width_taper + gap between tapered wg)/2
        taper: taper function
        cross_section:
        **kwargs: cross_section settings


    .. code::

                   length_mmi
                    <------>
                    ________
                   |        |
                __/          \__
            W1  __            __  E1
                  \          /_ _ _ _
                  |         | _ _ _ _| gap_mmi
                __/          \__
            W0  __            __  E0
                  \          /
                   |________|

                 <->
            length_taper

    """
    gf.snap.assert_on_2nm_grid(gap_mmi)
    x = gf.get_cross_section(cross_section)
    layer = x.layer

    component = gf.Component()
    w_mmi = width_mmi
    w_taper = width_taper

    taper = taper(
        length=length_taper,
        width1=width,
        width2=w_taper,
        cross_section=cross_section,
    )

    a = gap_mmi / 2 + width_taper / 2
    mmi = gf.components.rectangle(
        size=(length_mmi, w_mmi),
        layer=layer,
        centered=True,
    )

    ports = [
        gf.Port("o1", orientation=180, midpoint=(-length_mmi / 2, -a), width=w_taper),
        gf.Port("o2", orientation=180, midpoint=(-length_mmi / 2, +a), width=w_taper),
        gf.Port("o3", orientation=0, midpoint=(+length_mmi / 2, +a), width=w_taper),
        gf.Port("o4", orientation=0, midpoint=(+length_mmi / 2, -a), width=w_taper),
    ]

    mmi_section = component.add_ref(mmi)

    for port in ports:
        taper_ref = component << taper
        taper_ref.connect(port="o2", destination=port)
        component.add_port(name=port.name, port=taper_ref.ports["o1"])
        component.absorb(taper_ref)

    component.absorb(mmi_section)

    for layer, offset in zip(x.bbox_layers, x.bbox_offsets):
        points = gf.get_padding_points(
            component=c,
            default=0,
            bottom=offset,
            top=offset,
        )
        c.add_polygon(points, layer=layer)
    return component


if __name__ == "__main__":
    c = mmi2x2(gap_mmi=0.252)
    c.show()
    c.pprint()
