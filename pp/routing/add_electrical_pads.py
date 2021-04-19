from pp.cell import cell
from pp.component import Component
from pp.rotate import rotate
from pp.routing.route_pad_array import route_pad_array


@cell
def add_electrical_pads(
    component: Component, rotation: int = 180, **kwargs
) -> Component:
    """add compnent with top electrical pads and routes
    Args:
        component: Component,
        pad_spacing: float = 150.,
        pad: Callable = pad,
        fanout_length: Optional[int] = None,
        max_y0_optical: None = None,
        straight_separation: float = 4.0,
        bend_radius: float = 0.1,
        connected_port_list_ids: None = None,
        n_ports: int = 1,
        excluded_ports: List[Any] = [],
        pad_indices: None = None,
        route_filter: Callable = get_route_from_waypoints_electrical,
        port_name: str = "W",
        pad_rotation: int = -90,
        x_pad_offset: int = 0,
        port_labels: None = None,
        select_ports: Callable = select_electrical_ports,

    """

    c = Component(f"{component.name}_pad")
    cr = rotate(component=component, angle=rotation)

    elements, pads, _ = route_pad_array(
        component=cr,
        **kwargs,
    )

    c.add_ref(cr)
    for e in elements:
        c.add(e)
    for e in pads:
        c.add(e)

    for pname, p in cr.ports.items():
        if p.port_type == "optical":
            c.add_port(pname, port=p)

    return c.rotate(angle=-rotation)


if __name__ == "__main__":
    import pp

    # c.move((20, 50))
    # c = pp.components.cross(length=100, layer=pp.LAYER.M3, port_type="dc")
    # c = pp.components.mzi2x2(with_elec_connections=True)
    # c = add_electrical_pads(component=c, fanout_length=100)

    c = pp.components.straight_with_heater(length=200)
    c = add_electrical_pads(component=c)
    c.show()

    # print(cc.get_settings())
    # print(cc.ports)

    # ccc = pp.routing.add_fiber_array(component=cc)
    # pp.show(ccc)
