import functools

from gdsfactory import Component, cell


@cell
def wg(length: int = 3, layer: tuple[int, int] = (1, 0)) -> Component:
    """Dummy component for testing."""
    c = Component()
    width = 0.5
    w = width / 2
    c.add_polygon([(0, -w), (length, -w), (length, w), (0, w)], layer=layer)
    c.add_port(name="o1", center=(0, 0), width=width, orientation=180, layer=layer)
    c.add_port(name="o2", center=(length, 0), width=width, orientation=0, layer=layer)
    return c


@cell
def wg2(wg1=wg):
    """Dummy component for testing."""
    c = Component()
    w = wg1()
    w1 = c << w
    w1.rotate(90)
    c.copy_child_info(w)
    c.add_ports(w1.ports)
    return c


def test_set_name() -> None:
    c = wg(length=3, name="hi_there")
    assert c.name == "hi_there", c.name


@cell
def demo(length: int = 3, wg_width: float = 0.5) -> Component:
    """Demo Dummy cell."""
    c = Component()
    w = length
    h = wg_width
    points = [
        [-w / 2.0, -h / 2.0],
        [-w / 2.0, h / 2],
        [w / 2, h / 2],
        [w / 2, -h / 2.0],
    ]
    c.add_polygon(points)
    return c


def test_names() -> None:
    name_base = demo().name
    assert name_base.split("_")[0] == "demo", name_base

    demo2 = functools.partial(demo, length=3)
    c1 = demo2(length=3)
    c2 = demo(length=3)
    assert c1.name == c2.name, "{c1.name} != {c2.name}"

    c1 = demo(length=3, wg_width=0.5).name
    c2 = demo(wg_width=0.5, length=3).name
    assert c1 == c2, f"{c1} != {c2}"

    name_with_prefix = demo(prefix="hi").name
    assert name_with_prefix.split("_")[0] == "hi", name_with_prefix

    name_args = demo(3).name
    name_kwargs = demo(length=3).name
    assert name_args == name_kwargs, name_with_prefix
