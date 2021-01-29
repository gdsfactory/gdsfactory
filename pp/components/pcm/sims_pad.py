import pp


@pp.cell
def sims_pad(width=500, height=500, pad=100, layer=1):
    c = pp.Component()
    w = width
    h = height
    points = [
        [-w / 2.0, -h / 2.0],
        [-w / 2.0, h / 2],
        [w / 2, h / 2],
        [w / 2, -h / 2.0],
    ]
    c.add_polygon(points, layer=layer)
    w = width + 2 * pad
    h = height + 2 * pad
    points = [
        [-w / 2.0, -h / 2.0],
        [-w / 2.0, h / 2],
        [w / 2, h / 2],
        [w / 2, -h / 2.0],
    ]
    c.add_polygon(points, layer=pp.LAYER.PADDING)
    return c


if __name__ == "__main__":
    c = sims_pad()
    c.show()
