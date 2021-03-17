import pp
from pp.component import Component


def component_with_label_float() -> Component:
    c = pp.Component("ellipse_with_label")
    c << pp.components.ellipse()
    c.add_label(text="demo", position=(0.0, 0.0), layer=pp.LAYER.TEXT)
    return c


def component_with_label_int() -> Component:
    c = pp.Component("ellipse_with_label")
    c << pp.components.ellipse()
    c.add_label(text="demo", position=(0, 0), layer=pp.LAYER.TEXT)
    return c


def test_move_float_with_int() -> None:
    """ fixed """
    c = component_with_label_float()
    c.x = 10
    c.movex(10)


def test_move_int_with_float() -> None:
    """ needs fixing """
    c = component_with_label_int()
    c.x = 10.0
    c.movex(10.0)


if __name__ == "__main__":
    test_move_float_with_int()
    test_move_int_with_float()
    # c = component_with_label()
    # c.x = 10.0
    # c.show()
