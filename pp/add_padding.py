import numpy as np

import pp
from pp.container import container


@container
def add_padding(
    component, padding=50, x=None, y=None, layers=[pp.LAYER.PADDING], suffix="p"
):
    """ adds padding layers to a NEW component that has the same:
    - ports
    - settings
    - test_protocols and data_analysis_protocols

    as the old component
    """
    c = pp.Component(name=f"{component.name}_{suffix}")
    c << component
    c.ports = component.ports
    x = x or padding
    y = y or padding
    points = [
        [c.xmin - x, c.ymin - y],
        [c.xmax + x, c.ymin - y],
        [c.xmax + x, c.ymax + y],
        [c.xmin - x, c.ymax + y],
    ]
    for layer in layers:
        c.add_polygon(points, layer=layer)
    return c


@container
def add_padding_to_grid(
    component,
    grid_size=127,
    padding=10,
    bottom_padding=5,
    layers=[pp.LAYER.PADDING],
    suffix="p",
):
    """ returns component width a padding layer on each side
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

    x_padding -= padding
    y_padding -= padding

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
