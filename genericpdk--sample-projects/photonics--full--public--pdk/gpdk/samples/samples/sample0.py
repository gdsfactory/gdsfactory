"""Write GDS with hello world."""

import gdsfactory as gf


@gf.cell
def sample0_hello_world() -> gf.Component:
    c = gf.Component()
    ref1 = c.add_ref(gf.components.rectangle(size=(10, 10), layer=(1, 0)))
    ref2 = c.add_ref(gf.components.text("Hello", size=10, layer=(2, 0)))
    ref3 = c.add_ref(gf.components.text("world", size=10, layer=(2, 0)))
    ref1.xmax = ref2.xmin - 5
    ref3.xmin = ref2.xmax + 2
    ref3.rotate(90)
    return c
