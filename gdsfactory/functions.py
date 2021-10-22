"""All functions return a component so they are easy to pipe and compose."""
from functools import lru_cache

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.port import auto_rename_ports
from gdsfactory.types import Optional

cache = lru_cache


def add_port(component: Component, **kwargs) -> Component:
    """Returns Component with a new port."""
    component.add_port(**kwargs)
    return component


@cell
def rotate(
    component: Component,
    angle: int = 90,
) -> Component:
    """Returns rotated component inside a new component.

    Most times you just need to place a reference and rotate it.
    This rotate function just encapsulates the rotated reference into a new component.

    Args:
        component:
        angle: in degrees
    """
    component_new = Component()
    component_new.component = component
    ref = component_new.add_ref(component)
    ref.rotate(angle)
    component_new.add_ports(ref.ports)
    component_new.copy_child_info(component)
    return component_new


@cell
def move(
    component: Component,
    origin=(0, 0),
    destination=None,
    axis: Optional[str] = None,
) -> Component:
    component_new = Component()
    component_new.component = component
    ref = component_new.add_ref(component)
    ref.move(origin=origin, destination=destination, axis=axis)
    component_new.add_ports(ref.ports)
    component_new.copy_child_info(component)
    return component_new


def update_info(component: Component, **kwargs) -> Component:
    """Returns Component with updated info."""
    component.info.update(**kwargs)
    return component


__all__ = ("cache", "add_port", "rotate", "update_info", "auto_rename_ports")


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.mmi1x2()
    cr = c.rotate()
    cr.show()

    cm = move(c, destination=(20, 20))
    cm.show()

    # cr = rotate(component=c)
    # cr.show()
    # print(component_rotated)

    # component_rotated.pprint
    # component_netlist = component.get_netlist()
    # component.pprint_netlist()
