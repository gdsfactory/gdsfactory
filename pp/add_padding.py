import numpy as np

import pp


def add_padding(component, padding=50, layers=[pp.layer("padding")]):
    """ returns component width a padding layer on each side"""
    c = pp.Component(name=component.name + "_p")
    cr = c.add_ref(component)
    points = [
        [cr.xmin - padding, cr.ymin - padding],
        [cr.xmax + padding, cr.ymin - padding],
        [cr.xmax + padding, cr.ymax + padding],
        [cr.xmin - padding, cr.ymax + padding],
    ]
    for layer in layers:
        c.add_polygon(points, layer=layer)
    c.ports = cr.ports
    c.settings = component.settings
    return c


def add_padding_to_grid(
    component, grid_size=127, padding=10, bottom_padding=5, layers=[pp.layer("padding")]
):
    """ returns component width a padding layer on each side
    matches a minimum size
    grating couplers are at ymin
    """
    c = component
    c = pp.Component(
        name=c.name + "_p",
        settings=c.get_settings(),
        test_protocol=c.test_protocol,
        data_analysis_protocol=c.data_analysis_protocol,
    )
    cr = c.add_ref(component)

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
        [cr.xmin - x_padding / 2, cr.ymin - bottom_padding],
        [cr.xmax + x_padding / 2, cr.ymin - bottom_padding],
        [cr.xmax + x_padding / 2, cr.ymax + y_padding - bottom_padding],
        [cr.xmin - x_padding / 2, cr.ymax + y_padding - bottom_padding],
    ]
    for layer in layers:
        c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    c = pp.c.waveguide(length=128)
    cc = add_padding(c)
    # cc = add_padding_to_grid(c)
    print(cc.settings)
    print(cc.ports)
    pp.show(cc)
