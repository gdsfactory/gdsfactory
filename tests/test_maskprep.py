import numpy as np

import gdsfactory as gf
from gdsfactory.geometry.maskprep import fix_underplot


def test_fix_underplot():
    c1 = gf.Component("component_initial")
    c1 << gf.components.rectangle(size=(4, 4), layer=(1, 0))
    c1 << gf.components.rectangle(size=(2, 2), layer=(2, 0))
    slab = c1 << gf.components.rectangle(size=(2, 2), layer=(3, 0))
    slab.move((3, 1))

    c2 = gf.Component("component_clean")
    c2 = fix_underplot(
        component=c1,
        layers_extended=((2, 0), (3, 0)),
        layer_reference=(1, 0),
        distance=0.1,
    )

    # leaves reference untouched
    c1_WG = c1.extract(layers=((1, 0),))
    c2_WG = c2.extract(layers=((1, 0),))
    assert c1_WG.area() == c2_WG.area()
    # Shrinks (2,0)
    c1_SLAB150 = c1.extract(layers=((2, 0),))
    c2_SLAB150 = c2.extract(layers=((2, 0),))
    assert np.isclose(c1_SLAB150.area() - c2_SLAB150.area(), 0.39)
    # Splits (3,0)
    c1_SLAB90 = c1.extract(layers=((3, 0),)).get_polygons()
    c2_SLAB90 = c2.extract(layers=((3, 0),)).get_polygons()
    assert len(c2_SLAB90) - len(c1_SLAB90) == 1


if __name__ == "__main__":
    test_fix_underplot()
