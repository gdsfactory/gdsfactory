from gdsfactory.add_padding import get_padding_points
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bezier import bezier
from gdsfactory.cross_section import strip
from gdsfactory.port import auto_rename_ports
from gdsfactory.types import CrossSectionFactory, Float2


@cell
def bend_s(
    size: Float2 = (10.0, 2.0),
    nb_points: int = 99,
    with_cladding_box: bool = True,
    cross_section: CrossSectionFactory = strip,
    **kwargs
) -> Component:
    """S bend with bezier curve

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: in x and y direction
        nb_points: number of points
        with_cladding_box: square bounding box to avoid DRC errors
        cross_section: function
        kwargs: cross_section settings

    """
    dx, dy = size
    x = cross_section(**kwargs)
    width = x.info["width"]
    layer = x.info["layer"]

    c = Component()

    bend = bezier(
        width=width,
        control_points=[(0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy)],
        npoints=nb_points,
        layer=layer,
    )

    bend_ref = c << bend
    c.add_ports(bend_ref.ports)
    c.copy_child_info(bend)
    c.info.start_angle = bend.info.start_angle
    c.info.end_angle = bend.info.end_angle
    c.info.length = bend.info.length
    c.info.min_bend_radius = bend.info.min_bend_radius

    if with_cladding_box and x.info["layers_cladding"]:
        layers_cladding = x.info["layers_cladding"]
        cladding_offset = x.info["cladding_offset"]
        points = get_padding_points(
            component=c,
            default=0,
            bottom=cladding_offset,
            top=cladding_offset,
        )
        for layer in layers_cladding or []:
            c.add_polygon(points, layer=layer)

    auto_rename_ports(c)
    return c


if __name__ == "__main__":
    c = bend_s(width=1, layers_cladding=[(2, 0)])
    c = bend_s(size=[10, 2.5], layers_cladding=[(2, 0)])  # 10um bend radius
    c = bend_s(size=[20, 3], layers_cladding=[(2, 0)])  # 10um bend radius
    c.pprint()
    # c = bend_s_biased()
    # print(c.info["min_bend_radius"])
    c.show()
