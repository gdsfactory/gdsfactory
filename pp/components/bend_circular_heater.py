import pp
from pp.layers import LAYER
from pp.components.bend_circular import bend_circular
from pp.name import autoname
from pp.component import Component


@pp.port.deco_rename_ports
@autoname
def bend_circular_heater(radius: float = 10.0, wg_width: float = 0.5) -> Component:

    theta = -90
    start_angle = 0
    angle_resolution = 2.5
    heater_to_wg_distance = 1.2
    heater_width = 0.5

    return _bend_circular_heater(
        radius=radius,
        wg_width=wg_width,
        theta=theta,
        start_angle=start_angle,
        angle_resolution=angle_resolution,
        heater_to_wg_distance=heater_to_wg_distance,
        heater_width=heater_width,
    )


def _bend_circular_heater(
    radius: float = 10,
    wg_width: float = 0.5,
    theta: int = -90,
    start_angle: int = 0,
    angle_resolution: float = 2.5,
    heater_to_wg_distance: float = 1.2,
    heater_width: float = 0.5,
) -> Component:
    """ Creates an arc of arclength ``theta`` starting at angle ``start_angle``

    Args:
        radius
        width: of the waveguide
        theta: arc length
        start_angle:
        angle_resolution
    """
    component = Component()

    wg_bend = bend_circular(
        radius=radius,
        width=wg_width,
        theta=theta,
        start_angle=start_angle,
        angle_resolution=angle_resolution,
        layer=LAYER.WG,
    ).ref((0, 0))

    a = heater_to_wg_distance + wg_width / 2 + heater_width / 2

    heater_outer = bend_circular(
        radius=radius + a,
        width=heater_width,
        theta=theta,
        start_angle=start_angle,
        angle_resolution=angle_resolution,
        layer=LAYER.HEATER,
    ).ref((0, -a))

    heater_inner = bend_circular(
        radius=radius - a,
        width=heater_width,
        theta=theta,
        start_angle=start_angle,
        angle_resolution=angle_resolution,
        layer=LAYER.HEATER,
    ).ref((0, a))

    component.add(wg_bend)
    component.add(heater_outer)
    component.add(heater_inner)

    component.absorb(wg_bend)
    component.absorb(heater_outer)
    component.absorb(heater_inner)

    i = 0

    for device in [wg_bend, heater_outer, heater_inner]:
        for port in device.ports.values():
            component.ports["{}".format(i)] = port
            i += 1

    component.info["length"] = wg_bend.info["length"]
    component.radius = radius
    component.width = wg_width
    return component


if __name__ == "__main__":
    # from pprint import pprint
    c = bend_circular_heater()
    # pprint(lys._layers)

    # c = bend_circular()
    pp.show(c)
