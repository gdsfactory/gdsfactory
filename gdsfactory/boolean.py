from __future__ import annotations

from typing import TYPE_CHECKING

import kfactory as kf

from gdsfactory.component import Component, ComponentReference, boolean_operations

if TYPE_CHECKING:
    from gdsfactory.typings import ComponentOrReference, LayerSpec


def boolean(
    A: ComponentOrReference,
    B: ComponentOrReference,
    operation: str,
    layer: LayerSpec,
    layer1: LayerSpec | None = None,
    layer2: LayerSpec | None = None,
) -> Component:
    """Performs boolean operations between 2 Component or Instance objects.

    The `operation` parameter specifies the type of boolean operation to perform.
    Supported operations include {'not', 'and', 'or', 'xor', '-', '&', '|', '^'}:
      - `'|'` is equivalent to `'or'`
      - `'-'` is equivalent to `'not'`
      - `'&'` is equivalent to `'and'`
      - `'^'` is equivalent to `'xor'`

    Args:
        A: Component(/Reference) or list of Component(/References).
        B: Component(/Reference) or list of Component(/References).
        operation: {'not', 'and', 'or', 'xor', '-', '&', '|', '^'}.
        layer: Specific layer to put polygon geometry on.
        layer1: Specific layer to get polygons.
        layer2: Specific layer to get polygons.

    Returns: Component with polygon(s) of the boolean operations between
      the 2 input Components performed.

    .. plot::
      :include-source:

      import gdsfactory as gf

      c = gf.Component()
      c1 = c << gf.components.circle(radius=10)
      c2 = c << gf.components.circle(radius=9)
      c2.movex(5)

      c = gf.boolean(c1, c2, operation="xor")
      c.plot()

    """
    from gdsfactory import get_layer

    if operation not in boolean_operations:
        raise ValueError(
            f"Boolean operation {operation} not supported. Choose from {list(boolean_operations.keys())}"
        )

    c = Component()
    layer1 = layer1 or layer
    layer2 = layer2 or layer

    layer_index1 = get_layer(layer1)
    layer_index2 = get_layer(layer2)
    layer_index = get_layer(layer)

    if isinstance(A, kf.DKCell):
        ar = kf.kdb.Region(A.begin_shapes_rec(layer_index1))
    else:
        ar = get_ref_shapes(A, layer_index1)
    if isinstance(B, kf.DKCell):
        br = kf.kdb.Region(B.begin_shapes_rec(layer_index2))
    else:
        br = get_ref_shapes(B, layer_index2)
    c.shapes(layer_index).insert(boolean_operations[operation](ar, br))

    return c


def get_ref_shapes(ref: ComponentReference, layer_index: int) -> kf.kdb.Region:
    if ref.is_regular_array():
        base_inst_region = kf.kdb.Region(ref.cell.begin_shapes_rec(layer_index))

        br = kf.kdb.Region()
        for ia in range(ref.na):
            for ib in range(ref.nb):
                br.insert(
                    base_inst_region.transformed(
                        ref.cplx_trans * kf.kdb.ICplxTrans(ia * ref.a + ib * ref.b)
                    )
                )
    else:
        br = kf.kdb.Region(ref.cell.begin_shapes_rec(layer_index)).transformed(
            ref.cplx_trans
        )
    return br


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.components import bbox, coupler

    # c = gf.Component()
    # e2 = c << gf.components.ellipse(radii=(10, 6))
    # e3 = c << gf.components.ellipse(radii=(10, 4))
    # e3.dmovex(5)
    # c = boolean(A=e2, B=e3, operation="and")
    c0 = gf.Component()
    core = c0 << coupler()
    clad = c0 << bbox(core, layer=(2, 0))
    clad.dmovex(5)
    c = boolean(clad, core, operation="not", layer=(3, 0), layer1=(2, 0), layer2=(1, 0))
    c.show()
