import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def component_with_label_float() -> Component:
    c = gf.Component("component_with_label_float")
    c << gf.components.rectangle()
    c.add_label(text="demo", position=(0.0, 0.0), layer=gf.LAYER.TEXT)
    return c


@gf.cell
def component_with_label_int() -> Component:
    c = gf.Component("component_with_label_int")
    c << gf.components.rectangle()
    c.add_label(text="demo", position=(0, 0), layer=gf.LAYER.TEXT)
    return c


def test_move_float_with_int() -> Component:
    c = gf.Component("test_move_float_with_int")
    ref = c.add_ref(component_with_label_float())
    ref.movex(10)
    return c


def test_move_int_with_float() -> None:
    c = gf.Component("test_move_float_with_float")
    ref = c.add_ref(component_with_label_int())
    ref.movex(10)


if __name__ == "__main__":
    c = test_move_float_with_int()
    # test_move_int_with_float()
    # c = component_with_label()
    # c.x = 10.0
    c.show(show_ports=True)
