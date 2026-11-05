from __future__ import annotations

__all__ = ["blackbox"]

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.shapes.bbox import bbox_to_points
from gdsfactory.typings import ComponentSpec, LayerSpec


@gf.cell_with_module_name(tags=["containers"])
def blackbox(
    component: ComponentSpec = "mmi1x2",
    layer: LayerSpec = "FLOORPLAN",
) -> Component:
    """Returns a black box component with the same footprint and ports.

    Replaces the component geometry with a single rectangle covering its
    bounding box, keeping the original ports. Useful to hide proprietary
    layouts before sharing a GDS file with third parties. The real component
    can be swapped back in with Component.replace_instances.

    Args:
        component: component to hide.
        layer: layer to draw the bounding box rectangle on.
    """
    original = gf.get_component(component)
    bbox = original.dbbox()
    if bbox.empty():
        raise ValueError(
            f"Cannot create a blackbox from {original.name!r}: it has no geometry."
        )

    c = Component()
    c.add_polygon(bbox_to_points(bbox), layer=layer)

    # Only ports are copied. The original component is never instantiated and
    # its settings/info are not copied, so no geometry, hierarchy or
    # parameters leak into the resulting cell.
    c.add_ports(original.ports)
    return c


if __name__ == "__main__":
    gf.gpdk.PDK.activate()
    c = blackbox()
    c.draw_ports()
    c.show()
