from pp.cell import cell
from pp.component import Component
from pp.components.bend_circular import bend_circular
from pp.components.straight import straight
from pp.config import TECH
from pp.port import deco_rename_ports
from pp.types import Number

WIRE_WIDTH = 10.0


@deco_rename_ports
@cell
def wire(
    length: Number = 50.0, cross_section_settings=TECH.waveguide.metal_routing, **kwargs
) -> Component:
    """Straight wire.

    Args:
        length: straiht length
    """
    return straight(
        length=length, cross_section_settings=cross_section_settings, **kwargs
    )


@cell
def corner(
    radius: float = 5,
    angle: int = 90,
    npoints: int = 720,
    with_cladding_box: bool = False,
    cross_section_settings=TECH.waveguide.metal_routing,
    **kwargs
) -> Component:
    """90 degrees electrical bend

    Args:
        radius
        angle: angle of arc (degrees)
        npoints: Number of points used per 360 degrees
        cross_section_settings: settings for cross_section
        kargs: cross_section settings to extrude

    """
    return bend_circular(
        radius=radius,
        angle=angle,
        npoints=npoints,
        with_cladding_box=with_cladding_box,
        cross_section_settings=cross_section_settings,
        **kwargs
    )


if __name__ == "__main__":

    c = wire()
    c = corner()
    c.show()
