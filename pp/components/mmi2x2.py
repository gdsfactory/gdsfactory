import pp
from pp.add_padding import add_padding
from pp.component import Component
from pp.components.taper import taper as taper_function
from pp.config import TECH
from pp.cross_section import cross_section, get_waveguide_settings
from pp.types import ComponentFactory


@pp.cell_with_validator
def mmi2x2(
    width: float = TECH.component_settings.mmi2x2.width,
    width_taper: float = TECH.component_settings.mmi2x2.width_taper,
    length_taper: float = TECH.component_settings.mmi2x2.length_taper,
    length_mmi: float = TECH.component_settings.mmi2x2.length_mmi,
    width_mmi: float = TECH.component_settings.mmi2x2.width_mmi,
    gap_mmi: float = TECH.component_settings.mmi2x2.gap_mmi,
    taper: ComponentFactory = taper_function,
    waveguide: str = "strip",
    **kwargs
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
        waveguide_settings: settings for cross_section
        kwargs: overwrites waveguide_settings


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
    waveguide_settings = get_waveguide_settings(waveguide, **kwargs)
    x = cross_section(**waveguide_settings)
    cladding_offset = x.info["cladding_offset"]
    layers_cladding = x.info["layers_cladding"]
    layer = x.info["layer"]

    component = pp.Component()
    w_mmi = width_mmi
    w_taper = width_taper

    taper = taper(
        length=length_taper, width1=width, width2=w_taper, **waveguide_settings
    )

    a = gap_mmi / 2 + width_taper / 2
    mmi = pp.components.rectangle(
        size=(length_mmi, w_mmi),
        layer=layer,
        centered=True,
        ports={
            "E": [(+length_mmi / 2, -a, w_taper), (+length_mmi / 2, +a, w_taper)],
            "W": [(-length_mmi / 2, -a, w_taper), (-length_mmi / 2, +a, w_taper)],
        },
    )

    mmi_section = component.add_ref(mmi)

    for port_name, port in mmi_section.ports.items():
        taper_ref = component << taper
        taper_ref.connect(port="2", destination=port)
        component.add_port(name=port_name, port=taper_ref.ports["1"])
        component.absorb(taper_ref)

    component.simulation_settings = dict(port_width=1.5e-6)
    component.absorb(mmi_section)

    layers_cladding = layers_cladding or []
    if layers_cladding:
        add_padding(
            component,
            default=cladding_offset,
            right=0,
            left=0,
            top=cladding_offset,
            bottom=cladding_offset,
            layers=layers_cladding,
        )
    return component


if __name__ == "__main__":
    c = mmi2x2(waveguide="nitride")
    c.show()
    # print(c.get_optical_ports())
    c.pprint()
    c.show()
