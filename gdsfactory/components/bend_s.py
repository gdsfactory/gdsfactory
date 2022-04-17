import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bezier import bezier
from gdsfactory.cross_section import strip
from gdsfactory.port import auto_rename_ports
from gdsfactory.types import CrossSectionSpec, Float2


@cell
def bend_s(
    size: Float2 = (10.0, 2.0),
    nb_points: int = 99,
    cross_section: CrossSectionSpec = strip,
    **kwargs
) -> Component:
    """S bend with bezier curve

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: in x and y direction
        nb_points: number of points
        cross_section: function
        kwargs: cross_section settings

    """
    dx, dy = size
    x = gf.get_cross_section(cross_section, **kwargs)
    width = x.width
    layer = x.layer

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
    c.info["start_angle"] = bend.info["start_angle"]
    c.info["end_angle"] = bend.info["end_angle"]
    c.info["length"] = bend.info["length"]
    c.info["min_bend_radius"] = bend.info["min_bend_radius"]

    for layer, offset in zip(x.bbox_layers, x.bbox_offsets):
        points = get_padding_points(
            component=c,
            default=0,
            bottom=offset,
            top=offset,
        )
        c.add_polygon(points, layer=layer)

    auto_rename_ports(c)
    return c


if __name__ == "__main__":
    c = bend_s(width=1)
    c = bend_s(size=[10, 2.5])  # 10um bend radius
    c = bend_s(size=[20, 3])  # 10um bend radius
    c.pprint()
    # c = bend_s_biased()
    # print(c.info["min_bend_radius"])
    c.show()
