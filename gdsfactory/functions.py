"""All functions return a component so they are easy to pipe and compose."""
from functools import lru_cache, partial

import numpy as np

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.text_rectangular import text_rectangular
from gdsfactory.port import auto_rename_ports
from gdsfactory.types import (
    Anchor,
    ComponentFactory,
    ComponentOrFactory,
    Float2,
    List,
    Optional,
)

cache = lru_cache(maxsize=None)


def add_port(component: Component, **kwargs) -> Component:
    """Returns Component with a new port."""
    component.add_port(**kwargs)
    return component


@cell
def add_text(
    component: ComponentOrFactory,
    text: str = "",
    text_offset: Float2 = (0, 0),
    text_anchor: Anchor = "cc",
    text_factory: ComponentFactory = text_rectangular,
) -> Component:
    """Returns component inside a new component with text geometry.

    Args:
        component:
        text_offset: relative to component size info anchor. Defaults to center.
        text_prefix: for labels. For example. 'A' will produce 'A1', 'A2', ...
        text_anchors: relative to component (ce cw nc ne nw sc se sw center cc).
        text_factory: function to add text labels.
    """
    component = component() if callable(component) else component
    component_new = Component()
    component_new.component = component
    ref = component_new.add_ref(component)

    t = component_new << text_factory(text)
    t.move((np.array(text_offset) + getattr(ref.size_info, text_anchor)))

    component_new.add_ports(ref.ports)
    component_new.copy_child_info(component)
    return component_new


def add_texts(
    components: List[ComponentOrFactory],
    prefix: str = "",
    index0: int = 0,
    **kwargs,
) -> List[Component]:
    """Returns a list of Component with text labels.

    Args:
        components: list of components
        prefix: Optional prefix for the labels
        index0: defaults to 0 (0, for first component, 1 for second ...)

    keyword Args:
        text_offset: relative to component size info anchor. Defaults to center.
        text_prefix: for labels. For example. 'A' will produce 'A1', 'A2', ...
        text_anchors: relative to component (ce cw nc ne nw sc se sw center cc).
        text_factory: function to add text labels.
    """
    return [
        add_text(component, text=f"{prefix}{i+index0}", **kwargs)
        for i, component in enumerate(components)
    ]


@cell
def rotate(
    component: ComponentOrFactory,
    angle: int = 90,
) -> Component:
    """Returns rotated component inside a new component.

    Most times you just need to place a reference and rotate it.
    This rotate function just encapsulates the rotated reference into a new component.

    Args:
        component:
        angle: in degrees
    """
    component = component() if callable(component) else component
    component_new = Component()
    component_new.component = component
    ref = component_new.add_ref(component)
    ref.rotate(angle)
    component_new.add_ports(ref.ports)
    component_new.copy_child_info(component)
    return component_new


rotate90 = partial(rotate, angle=90)
rotate90n = partial(rotate, angle=-90)
rotate180 = partial(rotate, angle=180)


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
    "add_port",
    "add_text",
    "auto_rename_ports",
    "cache",
    "mirror",
    "move",
    "move_port_to_zero",
    "rotate",
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

    cm3 = add_text(c, "hi")
    cm3.show()

    # cr = rotate(component=c)
    # cr.show()
    # print(component_rotated)

    # component_rotated.pprint
    # component_netlist = component.get_netlist()
    # component.pprint_netlist()
