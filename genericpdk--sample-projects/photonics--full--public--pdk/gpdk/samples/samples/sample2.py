"""Write GDS with remove layers."""

import gdsfactory as gf


@gf.cell
def sample2_remove_layers() -> gf.Component:
    c = gf.Component()

    c.add_ref(gf.components.rectangle(size=(10, 1), layer=(1, 0)))
    c.add_ref(gf.components.rectangle(size=(10, 2), layer=(3, 0)))
    c.add_ref(gf.components.rectangle(size=(10, 3), layer=(2, 0)))
    c.flatten()

    assert len(c.layers) == 3
    c = c.remove_layers(layers=[(3, 0), (2, 0)])

    assert len(c.layers) == 1
    return c
