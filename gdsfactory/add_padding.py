from __future__ import annotations

from functools import partial

from kfactory import Instance

import gdsfactory as gf
from gdsfactory.component import Component, ComponentAllAngle, container
from gdsfactory.typings import ComponentSpec, Coordinate, LayerSpec


def get_padding_points(
    component: Component | Instance | ComponentAllAngle,
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
    c = component
    top = top if top is not None else default
    bottom = bottom if bottom is not None else default
    right = right if right is not None else default
    left = left if left is not None else default
    return [
        (c.dxmin - left, c.dymin - bottom),
        (c.dxmax + right, c.dymin - bottom),
        (c.dxmax + right, c.dymax + top),
        (c.dxmin - left, c.dymax + top),
    ]


def add_padding(
    component: ComponentSpec = "mmi2x2",
    layers: tuple[LayerSpec, ...] = ("PADDING",),
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
    layers: tuple[LayerSpec, ...] = ("PADDING",),
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
    component = gf.get_component(component)

    c = component
    top = abs(ysize - component.dysize) if ysize else 0
    right = abs(xsize - component.dxsize) if xsize else 0
    points = [
        (c.dxmin - left, c.dymin - bottom),
        (c.dxmax + right, c.dymin - bottom),
        (c.dxmax + right, c.dymax + top),
        (c.dxmin - left, c.dymax + top),
    ]

    for layer in layers:
        component.add_polygon(points, layer=layer)

    return component


add_padding_container = partial(container, function=add_padding)  # type: ignore
add_padding_to_size_container = partial(container, function=add_padding_to_size)  # type: ignore


if __name__ == "__main__":
    # test_container()

    # p = partial(add_padding, layers=((1, 0)))
    c = gf.components.straight(length=10)  # type: ignore
    c1 = add_padding_to_size_container(c, xsize=100, ysize=100)
    c2 = add_padding_to_size_container(c, xsize=100, ysize=100)
    print(c1.name)
    print(c2.name)
    c2.show()
