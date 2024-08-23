from __future__ import annotations

import gdsfactory as gf
from gdsfactory import partial


def test_partial_function_with_kwargs() -> None:
    s4 = partial(gf.components.rectangle, size=(4, 2))
    s4_args = partial(gf.components.rectangle, (4, 2))
    s6 = partial(gf.components.rectangle, size=(6, 2))

    c4 = s4()
    c4_args = s4_args()
    c6 = s6()

    assert c6.name != c4.name, f"{c6.name!r} must be different from {c4.name!r}"
    assert c4.name == c4_args.name, f"{c4.name!r} == {c4_args.name!r}"


def test_partial_function_without_kwargs() -> None:
    r1 = partial(gf.components.rectangle, size=(4, 2))
    r2 = partial(gf.components.rectangle, size=(4, 2))
    r3 = partial(gf.components.rectangle, (4, 2))

    c1 = r1()
    c2 = r2()
    c3 = r3()

    assert c1.name == c2.name == c3.name, f"{c1.name} == {c2.name} == {c3.name}"
