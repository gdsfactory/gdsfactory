import pp


def component_with_label():
    c = pp.Component("ellipse_with_label")
    c << pp.c.ellipse()
    c.add_label(text="demo", position=(0.0, 0.0), layer=pp.LAYER.TEXT)
    return c


def test_move():
    """ needs fixing """
    c = component_with_label()
    c.x = 10
    # c.x = 10
    c.movex(10)


if __name__ == "__main__":
    # test_move()
    c = component_with_label()
    c.x = 10.0
    pp.show(c)
