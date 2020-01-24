from numpy import floor
import pp


@pp.autoname
def via(width=0.7, height=0.7, period=2.0, clearance=1.0, layer=pp.LAYER.VIA1):
    """
    Square via
    """

    cmp = pp.Component()
    cmp.info["period"] = period
    cmp.info["clearance"] = clearance
    cmp.info["width"] = width
    cmp.info["height"] = height

    a = width / 2
    b = height / 2

    cmp.add_polygon([(-a, -b), (a, -b), (a, b), (-a, b)], layer=layer)

    return cmp


@pp.autoname
def via1(**kwargs):
    return via(layer=pp.LAYER.VIA1, **kwargs)


@pp.autoname
def via2(**kwargs):
    return via(layer=pp.LAYER.VIA2, **kwargs)


@pp.autoname
def via3(**kwargs):
    return via(layer=pp.LAYER.VIA3, **kwargs)


@pp.autoname
def tlm(
    width=11.0,
    height=11.0,
    layers=[pp.LAYER.M1, pp.LAYER.M2, pp.LAYER.M3],
    vias=[via2, via3],
):
    """
    Rectangular transition thru metal layers

    Args:
        name: component name
        width, height: rectangle parameters
        layers: layers on which to draw rectangles
        vias: vias to use to fill the rectangles

    Returns
        <pp.Component>
    """

    # assert len(layers) - 1 == len(vias), "tlm: There should be N layers for N-1 vias"

    a = width / 2
    b = height / 2
    rect_pts = [(-a, -b), (a, -b), (a, b), (-a, b)]

    cmp = pp.Component()
    # Add metal rectangles
    for layer in layers:
        cmp.add_polygon(rect_pts, layer=layer)

    # Add vias
    for via in vias:
        via = pp.call_if_func(via)

        w = via.info["width"]
        h = via.info["height"]
        c = via.info["clearance"]
        period = via.info["period"]

        nb_vias_x = (width - w - 2 * c) / period + 1
        nb_vias_y = (height - h - 2 * c) / period + 1

        nb_vias_x = int(floor(nb_vias_x))
        nb_vias_y = int(floor(nb_vias_y))

        cw = (width - (nb_vias_x - 1) * period - w) / 2
        ch = (height - (nb_vias_y - 1) * period - h) / 2

        x0 = -a + cw + w / 2
        y0 = -b + ch + h / 2

        for i in range(nb_vias_x):
            for j in range(nb_vias_y):
                cmp.add(via.ref(position=(x0 + i * period, y0 + j * period)))

    return cmp


if __name__ == "__main__":
    c = tlm()
    pp.show(c)
