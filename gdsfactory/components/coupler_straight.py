import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.types import ComponentFactory


@gf.cell
def coupler_straight(
    length: float = 10.0,
    gap: float = 0.27,
    straight: ComponentFactory = straight_function,
    **kwargs
) -> Component:
    """Coupler_straight with two parallel straights.

    Args:
        length: of straight
        gap: between straights
        straight: straight waveguide function
        kwargs:  cross_section settings
    """
    component = Component()

    straight_component = (
        straight(length=length, **kwargs) if callable(straight) else straight
    )

    top = component << straight_component
    bot = component << straight_component

    # bot.ymax = 0
    # top.ymin = gap

    top.movey(straight_component.info.width + gap)

    component.add_port("o1", port=bot.ports["o1"])
    component.add_port("o2", port=top.ports["o1"])
    component.add_port("o3", port=bot.ports["o2"])
    component.add_port("o4", port=top.ports["o2"])
    component.auto_rename_ports()
    return component


if __name__ == "__main__":
    c = coupler_straight(width=1)
    c.show(show_ports=True)
