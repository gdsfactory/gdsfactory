from typing import Optional

from pp.cell import cell
from pp.component import Component
from pp.components.straight import straight as straight_function
from pp.types import ComponentFactory, CrossSectionFactory


@cell
def coupler_straight(
    length: float = 10.0,
    gap: float = 0.27,
    straight: ComponentFactory = straight_function,
    snap_to_grid_nm: int = 1,
    cross_section_factory: Optional[CrossSectionFactory] = None,
    **cross_section_settings
) -> Component:
    """Coupler_straight with two parallel straights.

    Args:
        length: of straight
        gap: between straights
        straight: straight waveguide function
        snap_to_grid_nm
        cross_section_factory:
        **cross_section_settings
    """
    component = Component()

    straight_component = (
        straight(
            length=length,
            snap_to_grid_nm=snap_to_grid_nm,
            cross_section_factory=cross_section_factory,
            **cross_section_settings
        )
        if callable(straight)
        else straight
    )

    top = component << straight_component
    bot = component << straight_component

    # bot.ymax = 0
    # top.ymin = gap

    top.movey(straight_component.width + gap)

    component.add_port("W0", port=bot.ports["W0"])
    component.add_port("W1", port=top.ports["W0"])
    component.add_port("E0", port=bot.ports["E0"])
    component.add_port("E1", port=top.ports["E0"])
    return component


if __name__ == "__main__":
    c = coupler_straight(width=1)
    c.show(show_ports=True)
