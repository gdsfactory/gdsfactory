from typing import Any, Dict

from pp.cell import cell
from pp.component import Component
from pp.components.bend_euler import bend_euler
from pp.cross_section import cross_section
from pp.types import ComponentFactory


@cell
def coupler90bend(
    radius: float = 10.0,
    gap: float = 0.2,
    bend: ComponentFactory = bend_euler,
    cross_section_settings_inner: Dict[str, Any] = None,
    cross_section_settings_outer: Dict[str, Any] = None,
    **kwargs
) -> Component:
    r"""Returns 2 coupled bends.

    Args:
        radius: um
        gap: um
        bend: for bend
        layer: bend layer
        cross_section_settings_inner: for inner bend
        cross_section_settings_outer: for outer bend
        kwargs: cross_section_settings for both inner and outer


    .. code::

            r  N1 N0
            |   | |
            |  / /
            | / /
       W1____/ /
       W0_____/

    """

    c = Component()

    cross_section_settings_outer = cross_section_settings_outer or {}
    cross_section_settings_inner = cross_section_settings_inner or {}

    cross_section_settings_outer.update(**kwargs)
    cross_section_settings_inner.update(**kwargs)

    cross_section_inner = cross_section(**cross_section_settings_inner)
    cross_section_outer = cross_section(**cross_section_settings_outer)

    width = (
        cross_section_outer.info["width"] / 2 + cross_section_inner.info["width"] / 2
    )
    spacing = gap + width

    bend90_inner = bend(radius=radius, **cross_section_settings_inner)
    bend90_outer = bend(
        radius=radius + spacing,
        **cross_section_settings_outer,
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
    c = coupler90bend(radius=3, cross_section_settings_outer=dict(width=2))
    c.show()
