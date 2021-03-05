from typing import Iterable, Optional, Tuple

import numpy as np

import pp
from pp.component import Component
from pp.container import container


def get_padding_points(
    component: Component,
    default: float = 50.0,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    right: Optional[float] = None,
    left: Optional[float] = None,
) -> list:
    """Returns padding points for a component outline.

    Args:
        component
        default: default padding
        top: north padding
        bottom: south padding
        right: east padding
        left: west padding
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


@container
def add_padding(
    component: Component,
    layers: Tuple[Tuple[int, int], ...] = (pp.LAYER.PADDING),
    suffix: str = "p",
    **kwargs,
) -> Component:
    """Adds padding layers to a container.

    Returns the same ports on the component.

    Args:
        component
        layers: list of layers
        suffix for name
        default: default padding
        top: north padding
        bottom: south padding
        right: east padding
        left: west padding
    """

    c = pp.Component(name=f"{component.name}_{suffix}")
    c << component

    points = get_padding_points(component, **kwargs)
    for layer in layers:
        c.add_polygon(points, layer=layer)
    c.ports = component.ports
    return c


@container
def add_padding_to_grid(
    component: Component,
    grid_size: int = 127,
    x: int = 10,
    y: int = 10,
    bottom_padding: int = 5,
    layers: Iterable[Tuple[int, int]] = (pp.LAYER.PADDING,),
    suffix: str = "p",
) -> Component:
    """Returns component with padding layers on each side.

    New size is multiple of grid size
    """
    c = pp.Component(name=f"{component.name}_{suffix}")
    c << component
    c.ports = component.ports

    if c.size_info.height < grid_size:
        y_padding = grid_size - c.size_info.height
    else:
        n_grids = np.ceil(c.size_info.height / grid_size)
        y_padding = n_grids * grid_size - c.size_info.height

    if c.size_info.width < grid_size:
        x_padding = grid_size - c.size_info.width
    else:
        n_grids = np.ceil(c.size_info.width / grid_size)
        x_padding = n_grids * grid_size - c.size_info.width

    x_padding -= x
    y_padding -= y

    points = [
        [c.xmin - x_padding / 2, c.ymin - bottom_padding],
        [c.xmax + x_padding / 2, c.ymin - bottom_padding],
        [c.xmax + x_padding / 2, c.ymax + y_padding - bottom_padding],
        [c.xmin - x_padding / 2, c.ymax + y_padding - bottom_padding],
    ]
    for layer in layers:
        c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    c = pp.c.waveguide(length=128)
    cc = add_padding(component=c, layers=[(2, 0)], suffix="p")
    # cc = add_padding_to_grid(c, layers=[(2, 0)])
    # cc = add_padding_to_grid(c)
    # print(cc.settings)
    # print(cc.ports)
    cc.show()
