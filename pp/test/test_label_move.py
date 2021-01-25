import pp
from pp.component import Component


def test_label_move() -> Component:
    """ test that when we move a device its label also moves """
    c = pp.Component("ellipse_with_label")
    c << pp.c.ellipse()
    c.add_label(text="demo", position=(10, 0), layer=pp.LAYER.TEXT)
    c.movex(10)
    print(c.references)
    print(c.labels)
    # assert c.references[0].origin[0] == 10
    # assert c.labels[0].position[0] == 20
    return c


if __name__ == "__main__":
    c = test_label_move()
    pp.show(c)
