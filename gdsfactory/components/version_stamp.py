import datetime
import platform
from typing import Optional, Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.text import text
from gdsfactory.config import __version__
from gdsfactory.tech import LAYER


def pixel(size: int = 1.0, layer: Tuple[int, int] = LAYER.WG) -> Component:
    c = gf.Component()
    a = size / 2
    c.add_polygon([(a, a), (a, -a), (-a, -a), (-a, a)], layer)
    return c


@gf.cell
def qrcode(
    data: str = "mask01", psize: int = 1, layer: Tuple[int, int] = LAYER.WG
) -> Component:
    """Returns QRCode."""
    import qrcode

    pix = pixel(size=psize, layer=layer)
    q = qrcode.QRCode()
    q.add_data(data)
    matrix = q.get_matrix()
    c = gf.Component()
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value:
                pix_ref = pix.ref((i * psize, j * psize))
                c.add(pix_ref)
                c.absorb(pix_ref)
    return c


@gf.cell
def version_stamp(
    labels: Tuple[str, ...] = ("demo_label",),
    with_qr_code: bool = False,
    layer: Tuple[int, int] = LAYER.WG,
    pixel_size: int = 1,
    version: Optional[str] = __version__,
    text_size: int = 10,
) -> Component:
    """Component with module version and date.

    Args:
        labels: Iterable of labels
    """

    now = datetime.datetime.now()
    timestamp = "{:%Y-%m-%d %H:%M:%S}".format(now)
    short_stamp = "{:%y.%m.%d.%H.%M.%S}".format(now)

    c = gf.Component()
    if with_qr_code:
        data = f"{timestamp}/{platform.node()}"
        q = qrcode(layer=layer, data=data, psize=pixel_size).ref_center()
        c.add(q)
        c.absorb(q)

        x = q.size_info.width * 0.5 + 10

    else:
        x = 0

    txt_params = {"layer": layer, "justify": "left", "size": text_size}
    date = text(
        position=(x, text_size + 2 * pixel_size), text=short_stamp, **txt_params
    ).ref()
    c.add(date)
    c.absorb(date)

    if version:
        t = text(position=(x, 0), text=version, **txt_params).ref()
        c.add(t)
        c.absorb(t)

    for i, line in enumerate(labels):
        t = c << text(
            position=(x, -(i + 1) * (text_size + 2 * pixel_size)),
            text=line,
            **txt_params,
        )
        c.absorb(t)

    return c


if __name__ == "__main__":
    c = version_stamp(
        pixel_size=4,
        layer=LAYER.M1,
        with_qr_code=True,
        # text=["b1", "demo"],
        text_size=20,
    )
    c.show()
