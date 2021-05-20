import pp
from pp.component import Component
from pp.components.bend_circular import bend_circular
from pp.components.straight import straight
from pp.port import deco_rename_ports

WIRE_WIDTH = 10.0


@deco_rename_ports
@pp.cell_with_validator
def wire(length: float = 50.0, waveguide: str = "metal_routing", **kwargs) -> Component:
    """Straight wire.

    Args:
        length: straiht length
    """
    return straight(length=length, waveguide=waveguide, **kwargs)


@pp.cell_with_validator
def corner(
    radius: float = 5,
    angle: int = 90,
    npoints: int = 720,
    with_cladding_box: bool = False,
    waveguide: str = "metal_routing",
    **kwargs
) -> Component:
    """90 degrees electrical bend

    Args:
        radius
        angle: angle of arc (degrees)
        npoints: Number of points used per 360 degrees
        waveguide_settings: settings for cross_section
        kargs: cross_section settings to extrude

    """
    return bend_circular(
        radius=radius,
        angle=angle,
        npoints=npoints,
        with_cladding_box=with_cladding_box,
        waveguide=waveguide,
        **kwargs
    )


if __name__ == "__main__":

    c = wire()
    c = corner()
    c.show()
