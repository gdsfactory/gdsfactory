"""Based on phidl.geometry."""
from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentOrReference, LayerSpec


@gf.cell
def boolean(
    A: ComponentOrReference | tuple[ComponentOrReference, ...],
    B: ComponentOrReference | tuple[ComponentOrReference, ...],
    operation: str,
    precision: float = 1e-4,
    layer1: LayerSpec | None = None,
    layer2: LayerSpec | None = None,
    layer: LayerSpec = (1, 0),
) -> Component:
    """Performs boolean operations between 2 Component/Reference/list objects.

    ``operation`` should be one of {'not', 'and', 'or', 'xor', 'A-B', 'B-A', 'A+B'}.
    Note that 'A+B' is equivalent to 'or', 'A-B' is equivalent to 'not', and
    'B-A' is equivalent to 'not' with the operands switched

    You can also use gdsfactory.drc.boolean_klayout

    Args:
        A: Component(/Reference) or list of Component(/References).
        B: Component(/Reference) or list of Component(/References).
        operation: {'not', 'and', 'or', 'xor', 'A-B', 'B-A', 'A+B'}.
        precision: float Desired precision for rounding vertex coordinates.
        layer: Specific layer to put polygon geometry on.

    Returns: Component with polygon(s) of the boolean operations between
      the 2 input Components performed.

    Notes
    -----
    - 'A+B' is equivalent to 'or'.
    - 'A-B' is equivalent to 'not'.
    - 'B-A' is equivalent to 'not' with the operands switched.

    .. plot::
      :include-source:

      import gdsfactory as gf

      c1 = gf.components.circle(radius=10).ref()
      c2 = gf.components.circle(radius=9).ref()
      c2.movex(5)

      c = gf.geometry.boolean(c1, c2, operation="xor")
      c.plot()

    """
    c = Component()

    return c


def test_boolean() -> None:
    c = gf.Component()
    e1 = c << gf.components.ellipse()
    e2 = c << gf.components.ellipse(radii=(10, 6))
    e3 = c << gf.components.ellipse(radii=(10, 4))
    e3.movex(5)
    e2.movex(2)
    c = boolean(A=[e1, e3], B=e2, operation="A-B")
    assert len(c.polygons) == 2, len(c.polygons)


if __name__ == "__main__":
    # c = gf.Component()
    # e1 = c << gf.components.ellipse()
    # e2 = c << gf.components.ellipse(radii=(10, 6))
    # e3 = c << gf.components.ellipse(radii=(10, 4))
    # e3.movex(5)
    # e2.movex(2)
    # c = boolean(A=[e1, e3], B=e2, operation="A-B")

    n = 50
    c1 = gf.c.array(gf.c.circle(radius=10), columns=n, rows=n)
    c2 = gf.c.array(gf.c.circle(radius=9), columns=n, rows=n).ref()
    c2.movex(5)
    c = boolean(c1, c2, operation="xor")
    c.show()
