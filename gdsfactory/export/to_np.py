from __future__ import annotations

import numpy as np
import numpy.typing as npt

from gdsfactory.component import Component
from gdsfactory.typings import Floats, Layers


def to_np(
    component: Component,
    nm_per_pixel: int = 20,
    layers: Layers = ((1, 0),),
    values: Floats | None = None,
    pad_width: int = 1,
) -> npt.NDArray[np.float64]:
    """Returns a pixelated numpy array from Component polygons.

    Args:
        component: Component.
        nm_per_pixel: you can go from 20 (coarse) to 4 (fine).
        layers: to convert. Order matters (latter overwrite former).
        values: associated to each layer (defaults to 1).
        pad_width: padding pixels around the image.

    """
    import skimage.draw as skdraw

    pixels_per_um = (1 / nm_per_pixel) * 1e3

    xmin, ymin = component.bbox[0]
    xmax, ymax = component.bbox[1]

    shape = (
        int(np.ceil(xmax - xmin) * pixels_per_um),
        int(np.ceil(ymax - ymin) * pixels_per_um),
    )
    img = np.zeros(shape, dtype=float)
    layer_to_polygons = component.get_polygons_points(by="tuple")

    values = values or [1] * len(layers)

    for layer, value in zip(layers, values):
        if layer in layer_to_polygons:
            polygons = layer_to_polygons[layer]
            for polygon in polygons:
                r = polygon[:, 0] - xmin
                c = polygon[:, 1] - ymin
                rr, cc = skdraw.polygon(  # type: ignore
                    r * pixels_per_um, c * pixels_per_um, shape=shape
                )
                img[rr, cc] = value

    return np.pad(img, pad_width=pad_width)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from gdsfactory.components import bend_circular, straight

    c = straight()
    c = bend_circular()
    # i = to_np(c, nm_per_pixel=250)
    i = to_np(c, nm_per_pixel=20)
    c.show()
    plt.imshow(i.transpose(), origin="lower")  # type: ignore
    plt.colorbar()  # type: ignore
    plt.show()  # type: ignore
