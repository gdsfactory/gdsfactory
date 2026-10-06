from __future__ import annotations

__all__ = ["pixel", "qrcode", "version_stamp"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["pcms"])
def pixel(size: int = 1, layer: LayerSpec = "WG") -> Component:
    """Returns a square pixel, the building block of the QR code.

    Args:
        size: side length of the square, in um.
        layer: layer to use.
    """
    return cf.pixel(
        size=size,
        layer=layer,
    )


@gf.cell_with_module_name(tags=["pcms"])
def qrcode(data: str = "mask01", psize: int = 1, layer: LayerSpec = "WG") -> Component:
    """Returns QRCode.

    Args:
        data: string to encode.
        psize: pixel size.
        layer: layer to use.
    """
    return cf.qrcode(
        data=data,
        psize=psize,
        layer=layer,
    )


@gf.cell_with_module_name(tags=["pcms"])
def version_stamp(
    labels: tuple[str, ...] = ("demo_label",),
    with_qr_code: bool = False,
    layer: LayerSpec = "WG",
    pixel_size: int = 1,
    version: str | None = None,
    text_size: int = 10,
) -> Component:
    """Component with module version and date.

    Args:
        labels: Iterable of labels.
        with_qr_code: Whether to add a QR code with the date.
        layer: Layer to use.
        pixel_size: Pixel size.
        version: Version string.
        text_size: Text size.

    """
    return cf.version_stamp(
        labels=labels,
        with_qr_code=with_qr_code,
        layer=layer,
        pixel_size=pixel_size,
        version=version,
        text_size=text_size,
    )
