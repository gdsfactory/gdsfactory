from typing import Optional

from pp.cell import cell
from pp.component import Component
from pp.components.bend_euler import bend_euler
from pp.cross_section import CrossSectionFactory
from pp.tech import TECH_SILICON_C, Tech
from pp.types import ComponentFactory, Layer


@cell
def coupler90bend(
    radius: float = 10.0,
    gap: float = 0.2,
    bend: ComponentFactory = bend_euler,
    width: float = TECH_SILICON_C.wg_width,
    width_inner: Optional[float] = None,
    width_outer: Optional[float] = None,
    layer: Layer = TECH_SILICON_C.layer_wg,
    cross_section_factory_inner: Optional[CrossSectionFactory] = None,
    cross_section_factory_outer: Optional[CrossSectionFactory] = None,
    tech: Optional[Tech] = None,
    **kwargs
) -> Component:
    r"""Returns 2 coupled bends.

    Args:
        radius: um
        gap: um
        bend: for bend
        width: width of the bend
        width_inner: width of the inner bend
        width_outer: width of the outer bend
        layer: bend layer
        cross_section_factory_inner: for inner bend
        cross_section_factory_outer: for outer bend
        tech: Technology

    .. code::

            r  N1 N0
            |   | |
            |  / /
            | / /
       W1____/ /
       W0_____/

    """
    width_inner = width_inner or width
    width_outer = width_outer or width

    c = Component()
    bend90_inner = bend(
        radius=radius,
        width=width_inner,
        layer=layer,
        cross_section_factory=cross_section_factory_inner,
        tech=tech,
        **kwargs
    )
    spacing = gap + width_inner / 2 + width_outer / 2
    bend90_outer = bend(
        radius=radius + spacing,
        width=width_outer,
        layer=layer,
        cross_section_factory=cross_section_factory_outer,
        tech=tech,
        **kwargs
    )
    bend_inner_ref = c << bend90_inner
    bend_outer_ref = c << bend90_outer

    pbw = bend_inner_ref.ports["W0"]
    bend_inner_ref.movey(pbw.midpoint[1] + spacing)

    # This component is a leaf cell => using absorb
    c.absorb(bend_outer_ref)
    c.absorb(bend_inner_ref)

    c.add_port("N0", port=bend_outer_ref.ports["N0"])
    c.add_port("N1", port=bend_inner_ref.ports["N0"])
    c.add_port("W0", port=bend_outer_ref.ports["W0"])
    c.add_port("W1", port=bend_inner_ref.ports["W0"])

    return c


if __name__ == "__main__":
    c = coupler90bend(radius=3, width_inner=1)
    c.show()
