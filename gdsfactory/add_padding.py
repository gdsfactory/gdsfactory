from __future__ import annotations

from functools import partial
from typing import Any

import kfactory as kf

import gdsfactory as gf
from gdsfactory.component import Component, ComponentBase, ComponentReference, container
from gdsfactory.typings import ComponentSpec, Coordinate, LayerSpecs


def get_padding_points(
    component: ComponentBase | ComponentReference | kf.kcell.ProtoKCell[float, Any],
    default: float = 50.0,
    top: float | None = None,
    bottom: float | None = None,
    right: float | None = None,
    left: float | None = None,
) -> list[Coordinate]:
    """Returns padding points for a component outline.

    Args:
        component: to add padding.
        default: default padding in um.
        top: north padding in um.
        bottom: south padding in um.
        right: east padding in um.
        left: west padding in um.
    """
    # Assign padding values to fast local variables, avoid repeated attribute lookups
    if top is None:
        top_val = default
    else:
        top_val = top
    if bottom is None:
        bottom_val = default
    else:
        bottom_val = bottom
    if right is None:
        right_val = default
    else:
        right_val = right
    if left is None:
        left_val = default
    else:
        left_val = left

    # Pull all geometry into locals to avoid attribute lookups in each tuple construction
    xmin = component.xmin
    xmax = component.xmax
    ymin = component.ymin
    ymax = component.ymax

    xmin_left = xmin - left_val
    xmax_right = xmax + right_val
    ymin_bottom = ymin - bottom_val
    ymax_top = ymax + top_val

    # Construct points using local variables
    return [
        (xmin_left, ymin_bottom),
        (xmax_right, ymin_bottom),
        (xmax_right, ymax_top),
        (xmin_left, ymax_top),
    ]


def add_padding(
    component: ComponentSpec,
    layers: LayerSpecs = ("PADDING",),
    default: float = 50.0,
    top: float | None = None,
    bottom: float | None = None,
    right: float | None = None,
    left: float | None = None,
) -> Component:
    """Adds padding layers to component. Returns same component.

    Args:
        component: to add padding.
        layers: list of layers.
        default: default padding.
        top: north padding in um.
        bottom: south padding in um.
        right: east padding in um.
        left: west padding in um.
    """
    component = gf.get_component(component)

    points = get_padding_points(
        component,
        default=default,
        top=top,
        bottom=bottom,
        right=right,
        left=left,
    )
    for layer in layers:
        component.add_polygon(points, layer=layer)
    return component


def add_padding_to_size(
    component: ComponentSpec,
    layers: LayerSpecs = ("PADDING",),
    xsize: float | None = None,
    ysize: float | None = None,
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
    c = gf.get_component(component)

    top = abs(ysize - c.ysize) if ysize else 0
    right = abs(xsize - c.xsize) if xsize else 0
    points = [
        (c.xmin - left, c.ymin - bottom),
        (c.xmax + right, c.ymin - bottom),
        (c.xmax + right, c.ymax + top),
        (c.xmin - left, c.ymax + top),
    ]

    for layer in layers:
        c.add_polygon(points, layer=layer)

    return c


add_padding_container = partial(container, function=add_padding)
add_padding_to_size_container = partial(container, function=add_padding_to_size)


if __name__ == "__main__":
    # test_container()

    # p = partial(add_padding, layers=((1, 0)))
    c = gf.components.straight(length=10)
    c1 = add_padding_to_size_container(c, xsize=100, ysize=100)
    c2 = add_padding_to_size_container(c, xsize=100, ysize=100)
    print(c1.name)
    print(c2.name)
    c2.show()
