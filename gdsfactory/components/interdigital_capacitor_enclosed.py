from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.interdigital_capacitor import interdigital_capacitor
from gdsfactory.typings import LayerSpec


@gf.cell
def interdigital_capacitor_enclosed(
    enclosure_box: Sequence[Sequence[float | int]] = [[-200, -200], [200, 200]],
    cpw_dimensions: tuple[float | int, float | int] = (10, 6),
    gap_to_ground: float | int = 5,
    gap_layer: LayerSpec = "DEEPTRENCH",
    metal_layer: LayerSpec = "WG",
    **kwargs,
) -> Component:
    """Generates an interdigital capacitor surrounded by a ground plane and \
            coplanar waveguides with ports on both ends. \
            See for :func:`~interdigital_capacitor` for details.

    Note:
        ``finger_length=0`` effectively provides a plate capacitor.

    Args:
        enclosure_box: Bounding box dimensions for a ground metal enclosure.
        cpw_dimensions: Dimensions for the trace width and gap width of connecting coplanar waveguides.
        gap_to_ground: Size of gap from capacitor to ground metal.
        gap_layer: layer for trenching.
        metal_layer: layer for metalization.

    Keyword Args:
        fingers: total fingers of the capacitor.
        finger_length: length of the probing fingers.
        finger_gap: length of gap between the fingers.
        thickness: Thickness of fingers and section before the fingers.
    """
    c = Component()
    cap = interdigital_capacitor(**kwargs).ref_center()
    c.add(cap)

    gap = Component()
    for port in cap.get_ports_list():
        port2 = port.copy()

        # TODO very bad check, should do vector according to orientation
        direction = -1 if port.orientation > 0 else 1
        port2.move((30 * direction, 0))
        port2 = port2.flip()

        cpw_a, cpw_b = cpw_dimensions
        s0 = gf.Section(
            width=cpw_a, offset=0, layer=metal_layer, port_names=("in", "out")
        )
        s1 = gf.Section(width=cpw_b, offset=(cpw_a + cpw_b) / 2, layer=gap_layer)
        s2 = gf.Section(width=cpw_b, offset=-(cpw_a + cpw_b) / 2, layer=gap_layer)
        x = gf.CrossSection(sections=[s0, s1, s2])
        route = gf.routing.get_route(
            port,
            port2,
            cross_section=x,
        )
        c.add(route.references)

        # TODO check size parameters, now they're maybe approximate
        term = c << gf.components.bbox(
            [[0, 0], [cpw_b, cpw_a + 2 * cpw_b]], layer=gap_layer
        )
        if direction < 0:
            term.movex(-cpw_b)
        term.move(
            destination=route.ports[-1].move_copy(-1 * np.array([0, cpw_a / 2 + cpw_b]))
        )

        c.add_port(route.ports[-1])
        c.auto_rename_ports()

    gap.add_polygon(cap.get_polygon_enclosure(), layer=gap_layer)
    gap = gap.offset(gap_to_ground, layer=gap_layer)
    gap = gf.geometry.boolean(A=gap, B=c, operation="A-B", layer=gap_layer)

    ground = gf.components.bbox(bbox=enclosure_box, layer=metal_layer)
    ground = gf.geometry.boolean(
        A=ground, B=[c, gap], operation="A-B", layer=metal_layer
    )
    _ = c << ground
    return c.flatten()


if __name__ == "__main__":
    c = interdigital_capacitor_enclosed()
    c.show(show_ports=True)
