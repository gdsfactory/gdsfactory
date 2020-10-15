from typing import List, Union
import numpy as np

from omegaconf.listconfig import ListConfig
from phidl.device_layout import Layer
import pp
from pp.container import container
from pp.component import Component


@container
def add_padding(
    component: Component,
    padding: Union[float, int] = 50,
    x: None = None,
    y: None = None,
    layers: Union[List[ListConfig], List[Layer]] = [pp.LAYER.PADDING],
    suffix: str = "p",
) -> Component:
    """adds padding layers to a NEW component that has the same:
    - ports
    - settings
    - test_protocols and data_analysis_protocols

    as the old component
    """
    x = x if x is not None else padding
    y = y if y is not None else padding

    c = pp.Component(name=f"{component.name}_{suffix}")
    c << component
    c.ports = component.ports

    points = [
        [c.xmin - x, c.ymin - y],
        [c.xmax + x, c.ymin - y],
        [c.xmax + x, c.ymax + y],
        [c.xmin - x, c.ymax + y],
    ]
    for layer in layers:
        c.add_polygon(points, layer=layer)
    return c


def get_padding_points(
    component: Component,
    padding: Union[float, int] = 50,
    x: None = None,
    y: None = None,
) -> list:
    """ returns padding points for a component"""
    c = component
    x = x if x is not None else padding
    y = y if y is not None else padding
    return [
        [c.xmin - x, c.ymin - y],
        [c.xmax + x, c.ymin - y],
        [c.xmax + x, c.ymax + y],
        [c.xmin - x, c.ymax + y],
    ]


@container
def add_padding_to_grid(
    component,
    grid_size=127,
    x=10,
    y=10,
    bottom_padding=5,
    layers=[pp.LAYER.PADDING],
    suffix="p",
):
    """returns component width a padding layer on each side
    matches a minimum size
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
    print(cc.settings)
    print(cc.ports)
    pp.show(cc)
