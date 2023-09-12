import numpy as np

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, LayerSpec


@gf.cell
def layer_priority(
    component: ComponentSpec,
    layer_high_order: LayerSpec,
    layer_low_order: LayerSpec,
    remove_high_order: bool = False,
    **kwargs,
) -> gf.Component:
    """Returns new component after removing one layer from another.

    Args:
        component: spec.
        layer_high_order: layer used to etch.
        layer_low_order: layer etched into.
        remove_high_order: whether to also remove the high order layer polygons. \
                Useful if the higher order layer is purely logical.
        kwargs: keyword arguments for boolean difference operation.
    """
    c = gf.Component()

    component = gf.get_component(component)

    # Obtain component subsets
    layers_to_remove = (
        [gf.get_layer(layer_low_order), gf.get_layer(layer_high_order)]
        if remove_high_order
        else [gf.get_layer(layer_low_order)]
    )
    component_minus_layers = component.extract(
        [
            gf.get_layer(layer)
            for layer in component.get_layers()
            if gf.get_layer(layer) not in layers_to_remove
        ]
    )
    component_high_order = component.extract([layer_high_order])
    component_low_order = component.extract([layer_low_order])

    # Remove high priority from low priority
    component_high_order_removed_from_low_order = gf.geometry.boolean(
        A=component_low_order,
        B=component_high_order,
        operation="A-B",
        layer=layer_low_order,
        **kwargs,
    )

    # Place all the other layers
    a = c << component_minus_layers
    b = c << component_high_order_removed_from_low_order

    c.absorb(a)
    c.absorb(b)
    c.add_ports(ports=component.ports)
    c.copy_child_info(component)
    return c


def test_layer_priority() -> None:
    funky_cross_section = gf.cross_section.cross_section(
        width=0.5,
        layer="WG",
        sections=[
            gf.cross_section.Section(width=2, layer="N"),
            gf.cross_section.Section(width=2, layer="SLAB150"),
        ],
    )

    c = gf.components.coupler(cross_section=funky_cross_section)

    c_WG_before = c.extract([gf.get_layer("WG")])
    c_SLAB150_before = c.extract([gf.get_layer("SLAB150")])
    c_N_before = c.extract([gf.get_layer("N")])

    c_processed = layer_priority(
        component=c,
        layer_high_order="WG",
        layer_low_order="SLAB150",
        remove_high_order=False,
    )

    c_WG_after = c_processed.extract([gf.get_layer("WG")])
    c_SLAB150_after = c_processed.extract([gf.get_layer("SLAB150")])
    c_N_after = c_processed.extract([gf.get_layer("N")])

    assert np.isclose(c_WG_before.area(), c_WG_after.area(), atol=1e-3)
    assert np.isclose(c_N_before.area(), c_N_after.area(), atol=1e-3)
    assert c_SLAB150_before.area() > c_SLAB150_after.area()

    c_processed = layer_priority(
        component=c,
        layer_high_order="WG",
        layer_low_order="SLAB150",
        remove_high_order=True,
    )
    c_WG_after = c_processed.extract([gf.get_layer("WG")])

    assert c_WG_after.area() == 0


if __name__ == "__main__":
    funky_cross_section = gf.cross_section.cross_section(
        width=0.5,
        layer="WG",
        sections=[
            gf.cross_section.Section(width=2, layer="N"),
            gf.cross_section.Section(width=2, layer="SLAB150"),
        ],
    )

    c = gf.components.coupler(cross_section=funky_cross_section)

    c2 = layer_priority(
        component=c,
        layer_high_order="WG",
        layer_low_order="SLAB150",
        remove_high_order=False,
    )
    c2.show(show_ports=True)
