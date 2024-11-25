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
    extend_ports2, partial(gf.components.straight, width=0.5)
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

    # mzi500 = partial(gf.components.mzi, straight=straigth_extended1)
    # mzi900 = partial(gf.components.mzi, straight=straigth_extended2)
    # c500 = mzi500()
    # c900 = mzi900()

    assert c900.name != c500.name, f"{c500.name} must be different from {c900.name}"  # type: ignore


# def test_compose2():
#     """Ensures the second level of composed function gets a unique name.

#     FIXME! this one does not work

#     """
#     mzi500 = partial(gf.components.mzi, straight=straigth_extended3)
#     mzi900 = partial(gf.components.mzi, straight=straigth_extended2)

#     c500 = mzi500()
#     c900 = mzi900()

#     assert c900.name != c500.name, f"{c500.name} must be different from {c900.name}"


if __name__ == "__main__":
    test_compose1()
    # c = straigth_extended1()
    # c.show( )

    # mzi500 = partial(gf.components.mzi, straight=straigth_extended3)
    # mzi900 = partial(gf.components.mzi, straight=straigth_extended2)

    # c900 = mzi900()
    # c500 = mzi500()

    # c = gf.Component()
    # r500 = c << c500
    # r900 = c << c900
    # r900.dymin = r500.dymax + 10
    # c.show( )

    # assert c900.name != c500.name, f"{c500.name} must be different from {c900.name}"
