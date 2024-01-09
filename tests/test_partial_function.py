from __future__ import annotations

from functools import partial

import gdsfactory as gf


@gf.cell
def mmi_with_bend(
    mmi=gf.components.mmi1x2, bend=gf.components.bend_circular
) -> gf.Component:
    c = gf.Component()
    mmi = c << mmi()
    bend = c << bend()
    bend.connect("o2", mmi.ports["o2"])
    return c


def test_partial_function_with_kwargs() -> None:
    mmi400 = partial(gf.components.mmi1x2, width=0.4)
    mmi600 = partial(gf.components.mmi1x2, width=0.6)

    b400 = partial(gf.components.bend_circular, width=0.4)
    b600 = partial(gf.components.bend_circular, width=0.6)

    mmi_bend400 = partial(mmi_with_bend, mmi=mmi400, bend=b400)
    mmi_bend600 = partial(mmi_with_bend, mmi=mmi600, bend=b600)

    c400 = mmi_bend400()
    c600 = mmi_bend600()

    assert c600.name != c400.name, f"{c600.name} must be different from {c400.name}"


def test_partial_function_without_kwargs() -> None:
    r1 = partial(gf.components.rectangle, size=(4, 2))
    r2 = partial(gf.components.rectangle, size=(4, 2))
    r3 = partial(gf.components.rectangle, (4, 2))

    c1 = r1()
    c2 = r2()
    c3 = r3()

    assert c1.name == c2.name == c3.name, f"{c1.name} == {c2.name} == {c3.name}"
