"""All functions return a Component so you can easily pipe or compose them.

There are two types of functions:

- decorators: return the original component
- containers: return a new component

"""
from functools import lru_cache, partial

import numpy as np
from omegaconf import OmegaConf
from pydantic import validate_arguments

from gdsfactory import ComponentReference
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.straight import straight
from gdsfactory.components.text_rectangular import text_rectangular_multi_layer
from gdsfactory.port import auto_rename_ports
from gdsfactory.types import (
    Anchor,
    Axis,
    ComponentSpec,
    Float2,
    LayerSpec,
    List,
    Optional,
    Strs,
)

cache = lru_cache(maxsize=None)


def add_port(component: Component, **kwargs) -> Component:
    """Return Component with a new port."""
    component.add_port(**kwargs)
    return component


@cell
def add_text(
    component: ComponentSpec,
    text: str = "",
    text_offset: Float2 = (0, 0),
    text_anchor: Anchor = "cc",
    text_factory: ComponentSpec = text_rectangular_multi_layer,
) -> Component:
    """Return component inside a new component with text geometry.

    Args:
        component: component spec.
        text: text string.
        text_offset: relative to component anchor. Defaults to center (cc).
        text_anchor: relative to component (ce cw nc ne nw sc se sw center cc).
        text_factory: function to add text labels.

    """
    from gdsfactory.pdk import get_component

    component = get_component(component)
    component_new = Component()
    component_new.component = component
    ref = component_new.add_ref(component)

    t = component_new << text_factory(text)
    t.move(np.array(text_offset) + getattr(ref.size_info, text_anchor))

    component_new.add_ports(ref.ports)
    component_new.copy_child_info(component)
    return component_new


def add_texts(
    components: List[ComponentSpec],
    prefix: str = "",
    index0: int = 0,
    **kwargs,
) -> List[Component]:
    """Return a list of Component with text labels.

    Args:
        components: list of component specs.
        prefix: Optional prefix for the labels.
        index0: defaults to 0 (0, for first component, 1 for second ...).

    keyword Args:
        text_offset: relative to component size info anchor. Defaults to center.
        text_anchor: relative to component (ce cw nc ne nw sc se sw center cc).
        text_factory: function to add text labels.

    """
    return [
        add_text(component, text=f"{prefix}{i + index0}", **kwargs)
        for i, component in enumerate(components)
    ]


@cell
def rotate(
    component: ComponentSpec, angle: float = 90, recenter: bool = False
) -> Component:
    """Return rotated component inside a new component.

    Most times you just need to place a reference and rotate it.
    This rotate function just encapsulates the rotated reference into a new component.

    Args:
        component: spec.
        angle: to rotate in degrees.
        recenter: recenter component after rotating.

    """
    from gdsfactory.pdk import get_component

    component = get_component(component)
    component_new = Component()
    component_new.component = component
    ref = component_new.add_ref(component)

    origin_offset = ref.origin - np.array((ref.xmin, ref.ymin))

    ref.rotate(angle)

    if recenter:
        ref.move(
            origin=ref.center,
            destination=np.array((ref.xsize / 2, ref.ysize / 2)) - origin_offset,
        )

    component_new.add_ports(ref.ports)
    component_new.copy_child_info(component)
    return component_new


rotate90 = partial(rotate, angle=90)
rotate90n = partial(rotate, angle=-90)
rotate180 = partial(rotate, angle=180)


@cell
def mirror(
    component: ComponentSpec, p1: Float2 = (0, 1), p2: Float2 = (0, 0)
) -> Component:
    """Return new Component with a mirrored reference.

    Args:
        component: component spec.
        p1: first point to define mirror axis.
        p2: second point to define mirror axis.

    """
    from gdsfactory.pdk import get_component

    component = get_component(component)
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
    axis: Optional[Axis] = None,
) -> Component:
    """Return new Component with a moved reference to the original component.

    Args:
        component: to move.
        origin: of component.
        destination: Optional x, y.
        axis: x or y axis.

    """
    component_new = Component()
    component_new.component = component
    ref = component_new.add_ref(component)
    ref.move(origin=origin, destination=destination, axis=axis)
    component_new.add_ports(ref.ports)
    component_new.copy_child_info(component)
    return component_new


@cell
def transformed(ref: ComponentReference):
    """Returns flattened cell with reference transformations applied.

    Args:
        ref: the reference to flatten into a new cell.

    """
    c = Component()
    c.add(ref)
    c.copy_child_info(ref.ref_cell)
    c.add_ports(ref.ports)
    c.info["transformed_cell"] = ref.ref_cell.name
    return c.flatten()


def move_port_to_zero(component: Component, port_name: str = "o1"):
    """Return a container that contains a reference to the original component.

    The new component has port_name in (0, 0).

    """
    if port_name not in component.ports:
        raise ValueError(
            f"port_name = {port_name!r} not in {list(component.ports.keys())}"
        )
    return move(component, -component.ports[port_name].center)


def update_info(component: Component, **kwargs) -> Component:
    """Return Component with updated info."""
    component.info.update(**kwargs)
    return component


@validate_arguments
def add_settings_label(
    component: ComponentSpec = straight,
    layer_label: LayerSpec = (66, 0),
    settings: Optional[Strs] = None,
    ignore: Optional[Strs] = ("decorator",),
) -> Component:
    """Add a settings label to a component. Use it as a decorator.

    Args:
        component: spec.
        layer_label: for label.
        settings: list of settings to include. if None, adds all changed settings.
        ignore: list of settings to ignore.

    """
    from gdsfactory.pdk import get_component

    component = get_component(component)

    ignore = ignore or []
    settings = settings or component.settings.changed.keys()
    settings = set(settings) - set(ignore)

    d = {setting: component.get_setting(setting) for setting in settings}
    component.add_label(text=OmegaConf.to_yaml(d), layer=layer_label)
    return component


__all__ = (
    "add_port",
    "add_text",
    "add_settings_label",
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

    c = gf.components.mmi1x2(
        length_mmi=10,
        decorator=partial(add_settings_label, settings=["name", "length_mmi"]),
    )
    # c.show(show_ports=True)

    cr = rotate(component=c)
    cr.show()

    # cr = c.rotate()
    # cr.pprint()
    # cr.show()

    # cm = move(c, destination=(20, 20))
    # cm.show()

    # cm = mirror(c)
    # cm.show()

    # cm = c.mirror()
    # cm.show()

    # cm2 = move_port_to_zero(cm)
    # cm2.show()

    # cm3 = add_text(c, "hi")
    # cm3.show()

    # cr = rotate(component=c)
    # cr.show()
    # print(component_rotated)

    # component_rotated.pprint
    # component_netlist = component.get_netlist()
    # component.pprint_netlist()
