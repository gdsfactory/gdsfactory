from __future__ import annotations

from typing import List, Optional, Tuple

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, LayerSpec


def get_padding_points(
    component: Component,
    default: float = 50.0,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    right: Optional[float] = None,
    left: Optional[float] = None,
) -> List[float]:
    """Returns padding points for a component outline.

    Args:
        component: to add padding.
        default: default padding in um.
        top: north padding in um.
        bottom: south padding in um.
        right: east padding in um.
        left: west padding in um.
    """
    c = component
    top = top if top is not None else default
    bottom = bottom if bottom is not None else default
    right = right if right is not None else default
    left = left if left is not None else default
    return [
        [c.xmin - left, c.ymin - bottom],
        [c.xmax + right, c.ymin - bottom],
        [c.xmax + right, c.ymax + top],
        [c.xmin - left, c.ymax + top],
    ]


def add_padding(
    component: ComponentSpec = "mmi2x2",
    layers: Tuple[LayerSpec, ...] = ("PADDING",),
    **kwargs,
) -> Component:
    """Adds padding layers to component. Returns same component.

    Args:
        component: to add padding.
        layers: list of layers.

    keyword Args:
        default: default padding.
        top: north padding in um.
        bottom: south padding in um.
        right: east padding in um.
        left: west padding in um.
    """
    component = gf.get_component(component)

    points = get_padding_points(component, **kwargs)
    for layer in layers:
        component.add_polygon(points, layer=layer)
    return component


@cell
def add_padding_container(
    component: ComponentSpec,
    layers: Tuple[LayerSpec, ...] = ("PADDING",),
    **kwargs,
) -> Component:
    """Returns new component with padding added.

    Args:
        component: to add padding.
        layers: list of layers.

    Keyword Args:
        default: default padding in um.
        top: north padding in um.
        bottom: south padding in um.
        right: east padding in um.
        left: west padding in um.
    """
    component = gf.get_component(component)

    c = Component()
    c.component = component
    cref = c << component

    points = get_padding_points(component, **kwargs)
    for layer in layers:
        c.add_polygon(points, layer=layer)
    c.ports = cref.ports
    c.copy_child_info(component)
    return c


def add_padding_to_size(
    component: ComponentSpec,
    layers: Tuple[LayerSpec, ...] = ("PADDING",),
    xsize: Optional[float] = None,
    ysize: Optional[float] = None,
    left: float = 0,
    bottom: float = 0,
) -> Component:
    """Returns component with padding layers on each side.

    New size is multiple of grid size.

    Args:
        component: to add padding.
        layers: list of layers.
        xsize: x size to fill up in um.
        ysize: y size to fill up in um.
        left: left padding in um to fill up in um.
        bottom: bottom padding in um to fill up in um.
    """
    component = gf.get_component(component)

    c = component
    top = abs(ysize - component.ysize) if ysize else 0
    right = abs(xsize - component.xsize) if xsize else 0
    points = [
        [c.xmin - left, c.ymin - bottom],
        [c.xmax + right, c.ymin - bottom],
        [c.xmax + right, c.ymax + top],
        [c.xmin - left, c.ymax + top],
    ]

    for layer in layers:
        component.add_polygon(points, layer=layer)

    return component


@cell
def add_padding_to_size_container(
    component: ComponentSpec,
    layers: Tuple[LayerSpec, ...] = ("PADDING",),
    xsize: Optional[float] = None,
    ysize: Optional[float] = None,
    left: float = 0,
    bottom: float = 0,
) -> Component:
    """Returns new component with padding layers on each side. New size is.

    multiple of grid size.

    Args:
        component: to add padding.
        layers: list of layers.
        xsize: x size to fill up in um.
        ysize: y size to fill up in um.
        left: left padding in um to fill up in um.
        bottom: bottom padding in um to fill up in um.
    """
    component = gf.get_component(component)

    c = Component()
    cref = c << component

    top = abs(ysize - component.ysize) if ysize else 0
    right = abs(xsize - component.xsize) if xsize else 0
    points = [
        [cref.xmin - left, cref.ymin - bottom],
        [cref.xmax + right, cref.ymin - bottom],
        [cref.xmax + right, cref.ymax + top],
        [cref.xmin - left, cref.ymax + top],
    ]

    for layer in layers:
        c.add_polygon(points, layer=layer)

    c.ports = cref.ports
    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    # test_container()

    p = gf.partial(add_padding, layers=["SLAB150"])
    c = gf.components.straight(length=10, decorator=p)
    c.show()
