from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bezier import bezier
from gdsfactory.types import CrossSectionSpec, Float2


@cell
def bend_s(
    size: Float2 = (10.0, 2.0),
    nb_points: int = 99,
    cross_section: CrossSectionSpec = "strip",
    **kwargs
) -> Component:
    """Return S bend with bezier curve.

    stores min_bend_radius property in self.info['min_bend_radius']
    min_bend_radius depends on height and length

    Args:
        size: in x and y direction.
        nb_points: number of points.
        cross_section: spec.
        kwargs: cross_section settings.
    """
    c = Component()
    dx, dy = size

    bend = bezier(
        control_points=((0, 0), (dx / 2, 0), (dx / 2, dy), (dx, dy)),
        npoints=nb_points,
        cross_section=cross_section,
        **kwargs
    )
    bend_ref = c << bend
    c.add_ports(bend_ref.ports)
    c.copy_child_info(bend)
    return c


if __name__ == "__main__":
    c = bend_s(bbox_offsets=[0.5], bbox_layers=[(111, 0)])
    # c = bend_s(size=[10, 2.5])  # 10um bend radius
    # c = bend_s(size=[20, 3], cross_section="rib")  # 10um bend radius
    # c.pprint()
    # c = bend_s_biased()
    # print(c.info["min_bend_radius"])
    c.show(show_ports=True)
