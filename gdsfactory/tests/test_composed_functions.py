import toolz

import gdsfactory as gf

extend_ports1 = gf.partial(gf.c.extend_ports, length=1)
extend_ports2 = gf.partial(gf.c.extend_ports, length=10)


straigth_extended1 = toolz.compose(extend_ports1, gf.partial(gf.c.straight, width=0.5))
straigth_extended2 = toolz.compose(extend_ports2, gf.partial(gf.c.straight, width=0.9))

straigth_extended3 = toolz.compose(extend_ports2, gf.partial(gf.c.straight, width=0.5))


def test_compose1():
    """Ensures the first level of composed function gets a unique name"""
    mzi500 = gf.partial(gf.components.mzi, straight=straigth_extended1)
    mzi900 = gf.partial(gf.components.mzi, straight=straigth_extended2)

    c500 = mzi500()
    c900 = mzi900()

    assert c900.name != c500.name, f"{c500.name} must be different from {c900.name}"


def test_compose2():
    """Ensures the second level of composed function gets a unique name.

    FIXME! this one does not work

    """
    mzi500 = gf.partial(gf.components.mzi, straight=straigth_extended3)
    mzi900 = gf.partial(gf.components.mzi, straight=straigth_extended2)

    c500 = mzi500()
    c900 = mzi900()

    assert c900.name != c500.name, f"{c500.name} must be different from {c900.name}"


if __name__ == "__main__":
    mzi500 = gf.partial(gf.components.mzi, straight=straigth_extended3)
    mzi900 = gf.partial(gf.components.mzi, straight=straigth_extended2)

    c900 = mzi900()
    c500 = mzi500()

    c = gf.Component()
    r500 = c << c500
    r900 = c << c900
    r900.ymin = r500.ymax + 10
    c.show()

    assert c900.name != c500.name, f"{c500.name} must be different from {c900.name}"
