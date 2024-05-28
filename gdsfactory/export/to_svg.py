from __future__ import annotations

from gdsfactory.component import Component
from gdsfactory.technology import LayerStack, LayerViews
from gdsfactory.typings import Layer


def to_svg(
    component: Component,
    layer_views: LayerViews | None = None,
    layer_stack: LayerStack | None = None,
    exclude_layers: tuple[Layer, ...] | None = None,
    filename: str = "component.svg",
    scale: int = 1,
) -> None:
    """Write a 3D svg file from a component.

    Args:
        component: to extrude in 3D.
        layer_views: layer colors from Klayout Layer Properties file.
            Defaults to active PDK.layer_views.
        layer_stack: contains thickness and zmin for each layer.
            Defaults to active PDK.layer_stack.
        exclude_layers: layers to exclude.
        filename: svg filename.
        scale: scale for the svg.
    """
    from gdsfactory.pdk import get_layer_stack, get_layer_views

    layer_views = layer_views or get_layer_views()
    layer_stack = layer_stack or get_layer_stack()

    exclude_layers = exclude_layers or ()
    # layers = layer_views.layer_map.values()

    component_with_booleans = layer_stack.get_component_with_derived_layers(component)
    component_layers = component_with_booleans.layers
    xsize = component.dxsize
    ysize = component.dysize
    dcx, dcy = component.dcenter.x, component.dcenter.y
    dx, dy = dcx - xsize / 2, dcy - ysize / 2
    group_num = 1
    layer_to_polygons = component_with_booleans.get_polygons()

    with open(filename, "w+") as f:
        f.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
        f.write(
            f'<svg\n   width="{xsize * scale:0.6f}" \n   height="{ysize * scale:0.6f}"\n'
            '   version="1.1"\n'
            '   xmlns:svg="http://www.w3.org/2000/svg"\n'
            '   xmlns="http://www.w3.org/2000/svg">\n'
        )

        for level in layer_stack.layers.values():
            layer = level.layer

            if layer not in exclude_layers and layer in component_layers:
                zmin = level.zmin
                layer_view = layer_views.get_from_tuple(layer)
                color = layer_view.fill_color.as_hex(format="short")
                f.write('  <g id="layer%03i_datatype%03i">\n' % (layer[0], layer[1]))
                group_num += 1
                polygons = layer_to_polygons[layer]

                if zmin is not None and layer_view.visible:
                    for polygon in polygons:
                        poly_str = f'    <path style="fill:{color}"\n          d="'
                        for n, p in enumerate(polygon):
                            poly_str += "M " if n == 0 else "L "
                            poly_str += f"{(p[0] - dx) * scale:0.6f} {(-(p[1] - dy) + ysize) * scale:0.6f} "
                        poly_str += 'Z"/>\n'
                        f.write(poly_str)
                    f.write("  </g>\n")

        f.write("</svg>\n")


if __name__ == "__main__":
    import gdsfactory as gf

    # c = gf.components.taper_strip_to_ridge()
    # c = gf.Component()
    # c << gf.components.straight_heater_metal(length=40)
    # c << gf.c.rectangle(layer=(113, 0))
    # c = gf.components.mzi()
    # c = gf.components.taper_strip_to_ridge_trenches()
    c = gf.c.grating_coupler_elliptical_trenches()
    to_svg(c, scale=10)
    c.show()
