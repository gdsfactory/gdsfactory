from __future__ import annotations

from functools import partial

import toolz  # type: ignore

import gdsfactory as gf

extend_ports1 = partial(gf.components.extend_ports, length=1)
extend_ports2 = partial(gf.components.extend_ports, length=10)


straigth_extended1 = toolz.compose(  # type: ignore
    extend_ports1, partial(gf.components.straight, width=0.5)
)
straigth_extended2 = toolz.compose(  # type: ignore
    extend_ports2, partial(gf.components.straight, width=0.9)
)
straigth_extended3 = toolz.compose(  # type: ignore
    extend_ports2, partial(gf.components.straight, width=0.5, length=20)
)


def test_compose1() -> None:
    """Ensures the first level of composed function gets a unique name."""
    extend_ports1 = partial(gf.components.extend_ports, length=1)
    straigth_extended500 = toolz.compose(  # type: ignore
        extend_ports1, partial(gf.components.straight, width=0.5)
    )

    extend_ports2 = partial(gf.components.extend_ports, length=10)
    straigth_extended900 = toolz.compose(  # type: ignore
        extend_ports2, partial(gf.components.straight, width=0.9)
    )

    c500 = straigth_extended500()  # type: ignore
    c900 = straigth_extended900()  # type: ignore
    assert c900.name != c500.name, f"{c500.name} must be different from {c900.name}"  # type: ignore


def test_compose2() -> None:
    """Ensures the second level of composed function gets a unique name."""
    c1 = partial(gf.components.mzi, straight=straigth_extended1)()
    c3 = partial(gf.components.mzi, straight=straigth_extended3)()

    assert c1.name != c3.name, f"{c1.name} must be different from {c3.name}"
