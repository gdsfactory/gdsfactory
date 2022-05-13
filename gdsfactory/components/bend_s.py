import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bezier import bezier
from gdsfactory.port import auto_rename_ports
from gdsfactory.types import CrossSectionSpec, Float2


@cell
def bend_s(
    size: Float2 = (10.0, 2.0),
    nb_points: int = 99,
    with_bbox: bool = False,
    cross_section: CrossSectionSpec = "strip",
    **kwargs
) -> Component:
    """Return S bend with bezier curve
    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: in x and y direction
        nb_points: number of points
        with_bbox: box in bbox_layers and bbox_offsets to avoid DRC sharp edges.
        cross_section: function
        kwargs: cross_section settings

    """
    c = Component()
    dx, dy = size
    x = gf.get_cross_section(cross_section, **kwargs)
    width = x.width

    for name, section in x.aliases.items():
        width = section.width
        layer = section.layer
        bend = bezier(
            width=width,
            control_points=((0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy)),
            npoints=nb_points,
            layer=layer,
        )
        bend_ref = c << bend
        c.add_ports(bend_ref.ports, prefix=str(name))

    c.copy_child_info(bend)
    c.info["start_angle"] = bend.info["start_angle"]
    c.info["end_angle"] = bend.info["end_angle"]
    c.info["length"] = bend.info["length"]
    c.info["min_bend_radius"] = bend.info["min_bend_radius"]

    if x.info:
        c.info.update(x.info)

    if with_bbox:
        padding = []
        for layer, offset in zip(x.bbox_layers, x.bbox_offsets):
            points = get_padding_points(
                component=c,
                default=0,
                bottom=offset,
                top=offset,
            )
            padding.append(points)

        for layer, points in zip(x.bbox_layers, padding):
            c.add_polygon(points, layer=layer)

    auto_rename_ports(c)
    return c


if __name__ == "__main__":
    c = bend_s(width=1)
    # c = bend_s(size=[10, 2.5])  # 10um bend radius
    c = bend_s(size=[20, 3], cross_section="rib")  # 10um bend radius
    # c.pprint()
    # c = bend_s_biased()
    # print(c.info["min_bend_radius"])
    c.show()
