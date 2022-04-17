import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.component import Component
from gdsfactory.components.straight import straight as straight_function
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
    straight: CrossSectionSpec = straight_function,
    with_bbox: bool = True,
    cross_section: CrossSectionSpec = strip,
) -> Component:
    r"""Mmi 2x2.

    Args:
        width: input and output straight width.
        width_taper: interface between input straights and mmi region.
        length_taper: into the mmi region.
        length_mmi: in x direction.
        width_mmi: in y direction.
        gap_mmi: (width_taper + gap between tapered wg)/2.
        taper: taper function.
        straight: straight function.
        with_bbox: box in bbox_layers and bbox_offsets to avoid DRC sharp edges.
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

    c = gf.Component()
    w_mmi = width_mmi
    w_taper = width_taper

    taper = taper(
        length=length_taper,
        width1=width,
        width2=w_taper,
        cross_section=cross_section,
    )

    a = gap_mmi / 2 + width_taper / 2
    mmi = c << straight(length=length_mmi, width=w_mmi, cross_section=cross_section)

    ports = [
        gf.Port("o1", orientation=180, midpoint=(0, -a), width=w_taper),
        gf.Port("o2", orientation=180, midpoint=(0, +a), width=w_taper),
        gf.Port("o3", orientation=0, midpoint=(length_mmi, +a), width=w_taper),
        gf.Port("o4", orientation=0, midpoint=(length_mmi, -a), width=w_taper),
    ]

    for port in ports:
        taper_ref = c << taper
        taper_ref.connect(port="o2", destination=port)
        c.add_port(name=port.name, port=taper_ref.ports["o1"])
        c.absorb(taper_ref)

    if with_bbox:
        x = gf.get_cross_section(cross_section)
        padding = []
        for layer, offset in zip(x.bbox_layers, x.bbox_offsets):
            points = get_padding_points(
                component=c,
                default=0,
                bottom=offset,
                top=offset,
            )
            padding.append(points)

        for layer, points in zip(x.bbox_layers, padding):
            c.add_polygon(points, layer=layer)

    c.absorb(mmi)
    return c


if __name__ == "__main__":
    c = mmi2x2(gap_mmi=0.252, cross_section="rib")
    c.show()
    c.pprint()
