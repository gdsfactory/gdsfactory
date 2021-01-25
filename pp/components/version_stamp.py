import datetime
import platform
from typing import Tuple

import pp
from pp.component import Component
from pp.components.text import text as Text
from pp.config import conf
from pp.layers import LAYER


def pixel(size: int = 1.0, layer: Tuple[int, int] = LAYER.WG) -> Component:
    c = pp.Component()
    a = size / 2
    c.add_polygon([(a, a), (a, -a), (-a, -a), (-a, a)], layer)
    return c


@pp.cell
def qrcode(
    data: str = "gdsfactory", psize: int = 1, layer: Tuple[int, int] = LAYER.WG
) -> Component:
    """ A utility to render a list of pixels as a block """
    import qrcode

    pix = pixel(size=psize, layer=layer)
    q = qrcode.QRCode()
    q.add_data(data)
    matrix = q.get_matrix()
    c = pp.Component()
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value:
                pix_ref = pix.ref((i * psize, j * psize))
                c.add(pix_ref)
                c.absorb(pix_ref)
    return c


@pp.cell
def version_stamp(
    text=[], with_qr_code=False, layer=LAYER.WG, pixel_size=1, text_size=10
):
    """
    Includes module version, git hash and date

    .. plot::
      :include-source:

      import pp

      c = pp.c.version_stamp()
      pp.plotgds(c)

    """

    git_hash = conf.git_hash
    now = datetime.datetime.now()
    timestamp = "{:%Y-%m-%d %H:%M:%S}".format(now)
    short_stamp = "{:%y.%m.%d.%H.%M.%S}".format(now)

    c = pp.Component()
    if with_qr_code:
        data = "{}/{}/{}".format(git_hash, timestamp, platform.node())
        q = qrcode(layer=layer, data=data, psize=pixel_size).ref_center()
        c.add(q)
        c.absorb(q)

        x = q.size_info.width * 0.5 + 10

    else:
        x = 0

    txt_params = {"layer": layer, "justify": "left", "size": text_size}
    date = Text(
        position=(x, text_size + 2 * pixel_size), text=short_stamp, **txt_params
    ).ref()
    c.add(date)
    c.absorb(date)

    git_hash = Text(position=(x, 0), text=git_hash[:15], **txt_params).ref()
    c.add(git_hash)
    c.absorb(git_hash)

    for i, line in enumerate(text):
        text = c << Text(
            position=(x, -(i + 1) * (text_size + 2 * pixel_size)),
            text=line,
            **txt_params
        )
        c.absorb(text)

    return c


if __name__ == "__main__":
    print(conf.git_hash)
    c = version_stamp(
        pixel_size=4,
        layer=LAYER.M1,
        with_qr_code=True,
        # text=["b1", "demo"],
        text_size=20,
    )
    pp.show(c)
