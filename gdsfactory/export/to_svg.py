from __future__ import annotations

from typing import cast

from kfactory import LayerEnum

from gdsfactory.component import Component
from gdsfactory.technology import DerivedLayer, LayerStack, LayerViews, LogicalLayer
from gdsfactory.typings import LayerSpecs


def to_svg(
    component: Component,
    layer_views: LayerViews | None = None,
    layer_stack: LayerStack | None = None,
    exclude_layers: LayerSpecs | None = None,
    filename: str = "component.svg",
    scale: float = 1.0,
) -> None:
    """Write a 2D SVG file from a component.

    Args:
        component: The component to render.
        layer_views: Layer colors from Klayout Layer Properties file.
            Defaults to active `PDK.layer_views`.
        layer_stack: Contains thickness and zmin for each layer.
            Defaults to active `PDK.layer_stack`.
        exclude_layers: Layers to exclude from the SVG.
        filename: Output SVG filename.
        scale: Scaling factor for the SVG dimensions.
    """
    from gdsfactory.pdk import (
        get_active_pdk,
        get_layer,
        get_layer_stack,
        get_layer_views,
    )

    try:
        from shapely.geometry import Polygon
    except ImportError as e:
        print("You need to `pip install shapely` to use the `to_svg` function.")
        raise e

    layer_views = layer_views or get_layer_views()
    layer_stack = layer_stack or get_layer_stack()

    # Convert exclude_layers to layer indices for consistency
    exclude_layers = exclude_layers or ()
    exclude_layer_indices = [get_layer(layer) for layer in exclude_layers]

    # Prepare the component with boolean operations applied
    component_with_booleans = layer_stack.get_component_with_derived_layers(component)
    polygons_per_layer = component_with_booleans.get_polygons_points(merge=True)

    # Initialize SVG parameters
    xsize = component_with_booleans.xsize
    ysize = component_with_booleans.ysize
    dcx = component_with_booleans.center[0]
    dcy = component_with_booleans.center[1]
    dx, dy = dcx - xsize / 2, dcy - ysize / 2

    has_polygons = False

    with open(filename, "w") as f:
        # Write SVG header
        f.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
        f.write(
            f'<svg width="{xsize * scale}" height="{ysize * scale}" '
            'version="1.1" xmlns="http://www.w3.org/2000/svg">\n'
        )

        # Iterate through each layer in the stack
        for level in layer_stack.layers.values():
            layer = level.layer

            # Determine the layer tuple based on its type
            if isinstance(layer, LogicalLayer):
                assert isinstance(layer.layer, tuple | LayerEnum)
                layer_tuple = cast(tuple[int, int], tuple(layer.layer))
            elif isinstance(layer, DerivedLayer):
                assert level.derived_layer is not None
                assert isinstance(level.derived_layer.layer, tuple | LayerEnum)
                layer_tuple = cast(tuple[int, int], tuple(level.derived_layer.layer))
            else:
                raise ValueError(
                    f"Layer {layer!r} is not a DerivedLayer or LogicalLayer"
                )

            layer_index = get_layer(layer_tuple)

            # Skip excluded layers
            if layer_index in exclude_layer_indices:
                continue

            # Skip layers without polygons
            if layer_index not in polygons_per_layer:
                continue

            # Retrieve layer view properties
            layer_view = layer_views.get_from_tuple(layer_tuple)
            if not layer_view.visible or layer_view.fill_color is None:
                continue

            # Convert color to hex
            color_hex = layer_view.fill_color.as_hex(format="short")

            # Retrieve polygons for the current layer
            polygons = polygons_per_layer[layer_index]
            if not polygons:
                continue

            # Start SVG group for the layer
            f.write(
                f'  <g id="layer{layer_tuple[0]:03d}_datatype{layer_tuple[1]:03d}">\n'
            )

            for polygon_points in polygons:
                # Create a Shapely polygon for validation and processing
                polygon = Polygon(polygon_points)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)  # Attempt to fix invalid polygons

                if not polygon.is_valid or polygon.is_empty:
                    continue  # Skip invalid or empty polygons

                # Construct the SVG path string
                path_d = (
                    "M "
                    + " L ".join(
                        f"{(x - dx) * scale:.6f} {(ysize - (y - dy)) * scale:.6f}"
                        for x, y in polygon.exterior.coords
                    )
                    + " Z"
                )

                # Write the path to SVG
                f.write(
                    f'    <path style="fill:{color_hex};stroke:none;" d="{path_d}"/>\n'
                )
                has_polygons = True

            # Close SVG group
            f.write("  </g>\n")

        # Close SVG tag
        f.write("</svg>\n")

    if not has_polygons:
        raise ValueError(
            f"The component '{component.name}' does not contain any polygons in the specified layers "
            f"or the layers are excluded based on the active PDK '{get_active_pdk().name}'."
        )


if __name__ == "__main__":
    from gdsfactory.components import grating_coupler_elliptical_trenches

    # Example usage
    component = grating_coupler_elliptical_trenches()
    svg_filename = "grating_coupler.svg"
    to_svg(component, filename=svg_filename, scale=2.0)
    print(f"SVG file '{svg_filename}' has been created successfully.")
