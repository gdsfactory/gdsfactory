import gdsfactory as gf
from gdsfactory.component import Component


@gf.cell
def component_with_label() -> Component:
    c = gf.Component("component_with_label")
    c << gf.components.rectangle()
    c.add_label(text="demo", position=(0.0, 0.0), layer=gf.LAYER.TEXT)
    return c


def test_label_move() -> Component:
    """test that when we move references their label also move."""
    c = gf.Component("component_with_label_move")
    ref = c << gf.components.rectangle()
    ref.movex(10)
    assert ref.origin[0] == 10
    # assert ref.labels[0].center[0] == 10
    return c


if __name__ == "__main__":
    c = test_label_move()
    c.show(show_ports=True)
