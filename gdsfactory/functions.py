"""All functions return a component so they are easy to pipe and compose."""
from functools import lru_cache

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.port import auto_rename_ports
from gdsfactory.types import Float2, Optional

cache = lru_cache(maxsize=None)


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
def mirror(component: Component, p1: Float2 = (0, 1), p2: Float2 = (0, 0)) -> Component:
    """Returns mirrored component inside a new component.

    Args:
        p1: first point to define mirror axis
        p2: second point to define mirror axis
    """
    component_new = Component()
    component_new.component = component
    ref = component_new.add_ref(component)
    ref.mirror(p1=p1, p2=p2)
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
    """Returns a container that contains a reference to the original component."""
    component_new = Component()
    component_new.component = component
    ref = component_new.add_ref(component)
    ref.move(origin=origin, destination=destination, axis=axis)
    component_new.add_ports(ref.ports)
    component_new.copy_child_info(component)
    return component_new


def move_port_to_zero(component: Component, port_name: str = "o1"):
    """Returns a container that contains a reference to the original component.
    where the new component has port_name in (0, 0)
    """
    if port_name not in component.ports:
        raise ValueError(
            f"port_name = {port_name} not in {list(component.ports.keys())}"
        )
    return move(component, -component.ports[port_name].midpoint)


def update_info(component: Component, **kwargs) -> Component:
    """Returns Component with updated info."""
    component.info.update(**kwargs)
    return component


__all__ = (
    "cache",
    "add_port",
    "rotate",
    "auto_rename_ports",
    "move",
    "move_port_to_zero",
    "update_info",
)


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.mmi1x2()
    cr = c.rotate()
    cr.show()

    cm = move(c, destination=(20, 20))
    cm.show()

    cm = mirror(c)
    cm.show()

    cm = c.mirror()
    cm.show()

    cm2 = move_port_to_zero(cm)
    cm2.show()

    # cr = rotate(component=c)
    # cr.show()
    # print(component_rotated)

    # component_rotated.pprint
    # component_netlist = component.get_netlist()
    # component.pprint_netlist()
