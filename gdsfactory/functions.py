"""All functions return a Component so you can easily pipe or compose them.

There are two types of functions:

- decorators: return the original component
- containers: return a new component that contains the old one.

"""
from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from functools import lru_cache, partial
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from pydantic import validate_call

import gdsfactory as gf
from gdsfactory import ComponentReference
from gdsfactory.cell import cell_with_child, container
from gdsfactory.components.straight import straight
from gdsfactory.components.text_rectangular import text_rectangular_multi_layer
from gdsfactory.port import auto_rename_ports
from gdsfactory.typings import (
    Anchor,
    Axis,
    Component,
    ComponentSpec,
    Float2,
    LayerSpec,
    Strs,
)

cache = lru_cache(maxsize=None)


def _get_component_in_container(
    component: ComponentSpec, *args, **kwargs
) -> tuple[Component, Component, ComponentReference]:
    """Returns a new Component object that contains a reference to the original Component object.
    This allows effectively _modifying_ a Component after it has been created.

    Returns:
        A tuple containing the original Component object, the new Component object, and a reference to the original Component object.
    """
    from gdsfactory.pdk import get_component

    component = get_component(component, *args, **kwargs)
    component_new = Component()
    component_new.component = component
    ref = component_new.add_ref(component)
    return component, component_new, ref


def add_port(component: Component, **kwargs) -> Component:
    """Return Component with a new port."""
    component.add_port(**kwargs)
    return component


@cell_with_child
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
    component, component_new, ref = _get_component_in_container(component)

    t = component_new << text_factory(text)
    t.move(np.array(text_offset) + getattr(ref.size_info, text_anchor))

    component_new.add_ports(ref.ports)
    component_new.copy_child_info(component)
    return component_new


def add_texts(
    components: list[ComponentSpec],
    prefix: str = "",
    index0: int = 0,
    **kwargs,
) -> list[Component]:
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


@cell_with_child
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
    component, component_new, ref = _get_component_in_container(component)

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


@cell_with_child
def mirror(
    component: ComponentSpec, p1: Float2 = (0, 1), p2: Float2 = (0, 0)
) -> Component:
    """Return new Component with a mirrored reference.

    Args:
        component: component spec.
        p1: first point to define mirror axis.
        p2: second point to define mirror axis.

    """
    component, component_new, ref = _get_component_in_container(component)
    ref.mirror(p1=p1, p2=p2)
    component_new.add_ports(ref.ports)
    component_new.copy_child_info(component)
    return component_new


@cell_with_child
def move(
    component: Component,
    origin=(0, 0),
    destination=None,
    axis: Axis | None = None,
) -> Component:
    """Return new Component with a moved reference to the original component.

    Args:
        component: to move.
        origin: of component.
        destination: Optional x, y.
        axis: x or y axis.
    """
    component, component_new, ref = _get_component_in_container(component)
    ref.move(origin=origin, destination=destination, axis=axis)
    component_new.add_ports(ref.ports)
    component_new.copy_child_info(component)
    return component_new


@cell_with_child
def transformed(ref: ComponentReference) -> Component:
    """Returns flattened cell with reference transformations applied.

    Args:
        ref: the reference to flatten into a new cell.

    """
    from gdsfactory.component import copy_reference

    c = Component()
    ref = copy_reference(ref)
    c.add(ref)
    c.add_ports(ref.ports)
    c = c.flatten()
    c.copy_child_info(ref.ref_cell)
    c.info["transformed_cell"] = ref.ref_cell.name
    return c


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


@validate_call
def add_settings_label(
    component: ComponentSpec = straight,
    layer_label: LayerSpec = "TEXT",
    settings: Strs | None = None,
    ignore: Strs | None = ("decorator",),
    with_yaml_format: bool = True,
) -> Component:
    """Add a settings label to a component. Use it as a decorator.

    Args:
        component: spec.
        layer_label: for label.
        settings: list of settings to include. if None, adds all settings.
        ignore: list of settings to ignore.
        with_yaml_format: if True, uses yaml format, otherwise json.

    """
    from gdsfactory.pdk import get_component

    component = get_component(component)

    ignore = ignore or []
    settings = settings or dict(component.settings).keys()
    settings = set(settings) - set(ignore)

    d = dict(component.settings)
    d = {setting: d[setting] for setting in settings}
    text = OmegaConf.to_yaml(d) if with_yaml_format else json.dumps(d)
    component.add_label(text=text, layer=layer_label)
    return component


def add_marker_layer(
    component: ComponentSpec,
    marker_layer: LayerSpec,
    *,
    marker_label: str | None = None,
    layers_to_mark: list[LayerSpec] | None = None,
    flatten: bool = False,
) -> Component:
    """Adds a marker layer from the convex hull of the input component.
    Used as a decorator for `@gf.cell(decorator=partial(add_marker_layer, marker_layer=...)))`
    or as a decorator `c = gf.components.straight(decorator=partial(add_marker_layer, marker_layer=...))`

    Args:
        marker_layer: The marker layer.
        marker_label: An optional text label to add to the marker layer.
        layers_to_mark: Layers to use from component before taking convex hull. Defaults to all.
        flatten: Whether to flatten the component. Should be done only for elementary components.

    Returns:
        Same component with marker layer applied.
    """
    component = gf.get_component(component)

    if layers_to_mark:
        c = gf.Component()
        c.add_ref(component.extract(layers_to_mark))
    else:
        c = component
    polygon = c.get_polygons(as_shapely_merged=True)

    if not polygon.is_empty:
        component.add_polygon(polygon, layer=marker_layer)
        if marker_label:
            component.add_label(
                marker_label,
                position=(
                    (point := polygon.representative_point()).x,
                    point.y,
                ),
                layer=marker_layer,
            )
    else:
        warnings.warn(
            f"Could not add marker layer {marker_layer} to {component.name!r} because it is empty."
            f"Supplied {layers_to_mark=!r}.",
            stacklevel=2,
        )
    return component.flatten() if flatten else component


def change_keywords_in_nested_partials(
    func: gf.partial, config: Mapping[str, Any]
) -> gf.partial:
    """Change keywords in nested partials ``gf.partial or functools.partial``. Returns new partial.

    Args:
        func: Partialed function to change.
        config: Nested dictionary with the keywords to change.
            Key-value pairs correspond to function arguments in the partials.
    """

    if not config:
        return func

    if isinstance(func, gf.partial):
        keyword_args = dict(func.keywords)
        for key, value in config.items():
            if isinstance(keyword_args.get(key, None), gf.partial):
                # Recursively change keywords in nested partials
                keyword_args[key] = change_keywords_in_nested_partials(
                    keyword_args[key], value
                )
            else:
                keyword_args[key] = value

        return gf.partial(func.func, *func.args, **keyword_args)
    else:
        raise TypeError(f"{func=!r} is not a partial")


add_marker_layer_container = partial(container, function=add_marker_layer)


__all__ = (
    "add_marker_layer",
    "add_marker_layer_container",
    "add_port",
    "add_settings_label",
    "add_text",
    "auto_rename_ports",
    "cache",
    "change_keywords_in_nested_partials",
    "mirror",
    "move",
    "move_port_to_zero",
    "rotate",
    "update_info",
)

if __name__ == "__main__":
    c = gf.components.mmi1x2(
        length_mmi=10,
        decorator=partial(add_settings_label, settings=["name", "length_mmi"]),
    )
    # c.show(show_ports=True)

    # cr = rotate(component=c)
    # cr.show()

    cr = transformed(c.ref())
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
