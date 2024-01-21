from functools import partial

import gdstk

import gdsfactory as gf
from gdsfactory.typings import (
    Component,
    ComponentOrReference,
    ComponentSpec,
    Floats,
    LayerSpec,
    LayerSpecs,
)


def get_polygons_over_under(
    component: ComponentOrReference,
    layers: LayerSpecs,
    distances: Floats,
    use_union: bool = True,
    precision: float = 1e-4,
    join: str = "miter",
    tolerance: int = 2,
) -> list[gdstk.Polygon]:
    """Returns list polygons dilated and eroded by an offset.
    Cleans min width gap and acute angle DRC errors equal to distances.

    Args:
        component: Component containing polygons to offset.
        layers: list of layers to remove min gap errors.
        distances: Distance to offset polygons.
            By expanding and shrinking it merges polygons.
            If you want to fix min gaps of 0.5 use 0.5.
        precision: Desired precision for rounding vertex coordinates.
        join: {'miter', 'bevel', 'round'} Type of join used to create polygon offset
        tolerance: For miter joints, this number must be at least 2 represents the
          maximal distance in multiples of offset between new vertices and their
          original position before beveling to avoid spikes at acute joints. For
          round joints, it indicates the curvature resolution in number of
          points per full circle.
        layer: Specific layer to put polygon geometry on.
    Returns:
        Component containing a polygon(s) with the specified offset applied.
    """
    polygons = []

    for layer, distance in zip(layers, distances):
        layer = gf.get_layer(layer)
        if layer in component.layers:
            gds_layer, gds_datatype = layer
            polygons_to_offset = component.get_polygons(layer)
            p_dilated = gdstk.offset(
                polygons_to_offset,
                distance=+distance,
                join=join,
                tolerance=tolerance,
                precision=precision,
                use_union=use_union,
                layer=gds_layer,
                datatype=gds_datatype,
            )
            p = gdstk.offset(
                p_dilated,
                distance=-distance,
                join=join,
                tolerance=tolerance,
                precision=precision,
                use_union=use_union,
                layer=gds_layer,
                datatype=gds_datatype,
            )
            polygons.append(p)
    return polygons


@gf.cell_with_child
def over_under(
    component: ComponentSpec,
    layers: LayerSpecs,
    remove_original: bool = False,
    **kwargs,
) -> Component:
    """Returns cleaned component.

    Args:
        component:
        layers: list of layers.
        remove_original: remove original layers.
    """
    c = Component()
    _component = gf.get_component(component)

    ref = c << _component
    if remove_original:
        c = c.remove_layers(layers=layers)
    p = get_polygons_over_under(component=_component, layers=layers, **kwargs)
    c.add(p)
    c.copy_child_info(_component)
    c.add_ports(ref.ports)
    return c


over_under_remove_original = partial(over_under, remove_original=True)


def get_polygons_underplot(
    component: ComponentOrReference,
    layers_extended: LayerSpecs,
    layer_reference: LayerSpec,
    distance: Floats,
    keep_overplot: bool = True,
    use_union: bool = True,
    precision: float = 1e-4,
    join: str = "miter",
    tolerance: int = 2,
) -> list[gdstk.Polygon]:
    """Returns polygons on "layers_extended" .
    Cleans min width gap and acute angle DRC errors equal to distances.

    Args:
        component: Component containing polygons to offset.
        layers: list of layers to remove min gap errors.
        distances: Distance to offset polygons.
            By expanding and shrinking it merges polygons.
            If you want to fix min gaps of 0.5 use 0.5.
        precision: Desired precision for rounding vertex coordinates.
        join: {'miter', 'bevel', 'round'} Type of join used to create polygon offset
        tolerance: For miter joints, this number must be at least 2 represents the
          maximal distance in multiples of offset between new vertices and their
          original position before beveling to avoid spikes at acute joints. For
          round joints, it indicates the curvature resolution in number of
          points per full circle.
        layer: Specific layer to put polygon geometry on.
    Returns:
        Component containing a polygon(s) with the specified offset applied.
    """
    polygons = []

    # Acquire and offset the reference polygons
    layer_reference = gf.get_layer(layer_reference)
    if layer_reference in component.layers:
        gds_layer, gds_datatype = layer_reference
        reference_polygons = component.get_polygons(layer_reference)
        reference_polygons_shrunk = gdstk.offset(
            reference_polygons,
            distance=-distance,
            join=join,
            tolerance=tolerance,
            precision=precision,
            use_union=use_union,
            layer=gds_layer,
            datatype=gds_datatype,
        )

    # For each other layer, trim the polygons
    for layer in layers_extended:
        layer = gf.get_layer(layer)
        if layer in component.layers:
            gds_layer, gds_datatype = layer
            polygons_to_trim = component.get_polygons(layer)
            p = gdstk.boolean(
                polygons_to_trim,
                reference_polygons_shrunk,
                operation="and",
                precision=precision,
                layer=gds_layer,
                datatype=gds_datatype,
            )
            polygons.append(p)
            # Optionally add portion of the polygons outside the reference
            if keep_overplot:
                p = gdstk.boolean(
                    polygons_to_trim,
                    reference_polygons,
                    operation="not",
                    precision=precision,
                    layer=gds_layer,
                    datatype=gds_datatype,
                )
                polygons.append(p)
    return polygons


@gf.cell_with_child
def fix_underplot(
    component: ComponentSpec,
    layers_extended: LayerSpecs,
    layer_reference: LayerSpecs,
    distance: Floats,
    keep_overplot: bool = True,
    **kwargs,
) -> Component:
    """Returns component with polygon underplots.

    Args:
        component:
        layers_extended: list of layers to change.
        layer_reference: layers to underplot w.r.t. (union of these layers is considered)
        distances: list of floats, amount to overplot by
        remove_original: remove original layers.
    """
    c = Component()
    _component = gf.get_component(component)

    ref = c << _component
    c = c.remove_layers(layers=layers_extended)
    p = get_polygons_underplot(
        component=_component,
        layers_extended=layers_extended,
        layer_reference=layer_reference,
        distance=distance,
        keep_overplot=keep_overplot,
        **kwargs,
    )
    c.add(p)
    c.copy_child_info(_component)
    c.add_ports(ref.ports)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    # over_under_slab = partial(over_under, layers=((2, 0)), distances=(0.5,))
    # c = gf.components.coupler_ring(
    #     cladding_layers=((2, 0)),
    #     cladding_offsets=(0.2,),
    #     decorator=over_under_slab,
    # )
    # over_under_slab_remove_original = partial(
    #     over_under_remove_original, layers=((2, 0)), distances=(0.5,)
    # )
    # c = gf.components.coupler_ring(
    #     cladding_layers=((2, 0)),
    #     cladding_offsets=(0.2,),
    #     decorator=over_under_slab_remove_original,
    # )
    # get_polygons_over_under_slab = partial(
    #     get_polygons_over_under, layers=((2, 0)), distances=(0.5,)
    # )

    # c = gf.Component("component_clean")
    # ref = c << gf.components.coupler_ring(
    #     cladding_layers=((2, 0)),
    #     cladding_offsets=(0.2,),  # decorator=over_under_slab_decorator
    # )
    # polygons = get_polygons_over_under_slab(ref)
    # c.add(polygons)

    # Make component requiring overplot
    c1 = gf.Component("component_initial")
    c1 << gf.components.rectangle(size=(4, 4), layer="WG")
    c1 << gf.components.rectangle(size=(2, 2), layer="SLAB150")
    slab = c1 << gf.components.rectangle(size=(2, 2), layer="SLAB90")
    slab.move((3, 1))
    c1.show(show_ports=True)

    c2 = gf.Component("component_clean")
    c2 = fix_underplot(
        component=c1,
        layers_extended=("SLAB150", "SLAB90"),
        layer_reference="WG",
        distance=0.1,
        keep_overplot=False,
    )
    c2.show(show_ports=True)
