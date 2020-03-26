import numpy as np

import pp


def add_padding(component, padding=50, x=None, y=None, layers=[pp.LAYER.PADDING]):
    """ adds padding layers to component"""
    c = component
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


def add_padding_to_grid(
    component, grid_size=127, padding=10, bottom_padding=5, layers=[pp.LAYER.PADDING]
):
    """ returns component width a padding layer on each side
    matches a minimum size
    grating couplers are at ymin
    """
    c = component

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
    # cc = add_padding(c, layers=[(2, 0)])
    cc = add_padding_to_grid(c, layers=[(2, 0)])
    # cc = add_padding_to_grid(c)
    print(cc.settings)
    print(cc.ports)
    pp.show(cc)
