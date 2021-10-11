from typing import Optional

import numpy as np
import skimage.draw as skdraw

from gdsfactory.component import Component
from gdsfactory.types import Floats, Layers


def to_np(
    component: Component,
    nm_per_pixel: int = 20,
    layers: Layers = ((1, 0),),
    values: Optional[Floats] = None,
    pad_width: int = 1,
) -> np.ndarray:
    """Returns a pixelated numpy array from Component polygons.

    Args:
        component: Component
        nm_per_pixel: you can go from 20 (coarse) to 4 (fine)
        layers: to convert. Order matters (latter overwrite former)
        values: associated to each layer (defaults to 1)
        pad_width: padding pixels around the image

    """
    pixels_per_um = (1 / nm_per_pixel) * 1e3
    xmin, ymin = component.bbox[0]
    xmax, ymax = component.bbox[1]
    shape = (
        int(np.ceil(xmax - xmin) * pixels_per_um),
        int(np.ceil(ymax - ymin) * pixels_per_um),
    )
    img = np.zeros(shape, dtype=float)
    layer_to_polygons = component.get_polygons(by_spec=True, depth=None)

    values = values or [1] * len(layers)

    for layer, value in zip(layers, values):
        if layer in layer_to_polygons:
            polygons = layer_to_polygons[layer]
            for polygon in polygons:
                r = polygon[:, 0] - xmin
                c = polygon[:, 1] - ymin
                rr, cc = skdraw.polygon(
                    r * pixels_per_um, c * pixels_per_um, shape=shape
                )
                img[rr, cc] = value

    img_with_padding = np.pad(img, pad_width=pad_width)
    return img_with_padding


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import gdsfactory as gf

    c = gf.c.straight()
    c = gf.c.bend_circular(layers_cladding=[gf.LAYER.WGCLAD], cladding_offset=3.0)
    # i = to_np(c, nm_per_pixel=250)
    i = to_np(c, nm_per_pixel=20)
    c.show()
    plt.imshow(i.transpose(), origin="lower")
    plt.colorbar()
    plt.show()
