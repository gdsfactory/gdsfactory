"""Read component from a numpy.ndarray."""
from __future__ import annotations

import numpy as np

from gdsfactory.cell import cell_without_validator
from gdsfactory.component import Component
from gdsfactory.geometry.boolean import boolean


def compute_area_signed(pr) -> float:
    """Return the signed area enclosed by a ring using the linear time.

    algorithm at http://www.cgafaq.info/wiki/Polygon_Area. A value >= 0
    indicates a counter-clockwise oriented ring.

    """
    xs, ys = map(list, zip(*pr))
    xs.append(xs[1])
    ys.append(ys[1])
    return sum(xs[i] * (ys[i + 1] - ys[i - 1]) for i in range(1, len(pr))) / 2.0


@cell_without_validator
def from_np(
    ndarray: np.ndarray,
    nm_per_pixel: int = 20,
    layer: tuple[int, int] = (1, 0),
    threshold: float = 0.99,
    invert: bool = True,
) -> Component:
    """Returns Component from a np.ndarray.

    Extracts contours skimage.measure.find_contours using `threshold`.

    Args:
        ndarray: 2D ndarray representing the device layout.
        nm_per_pixel: scale_factor.
        layer: layer tuple to output gds.
        threshold: value along which to find contours in the array.
        invert: invert the mask.

    """
    from skimage import measure

    c = Component()
    d = Component()
    ndarray = np.pad(ndarray, 2)
    contours = measure.find_contours(ndarray, threshold)
    assert len(contours) > 0, (
        f"no contours found for threshold = {threshold}, maybe you can reduce the"
        " threshold"
    )

    for contour in contours:
        area = compute_area_signed(contour)
        points = contour * 1e-3 * nm_per_pixel
        if area < 0:
            c.add_polygon(points, layer=layer)
        else:
            d.add_polygon(points, layer=layer)

    return boolean(c, d, operation="not", layer=layer) if invert else d


@cell_without_validator
def from_image(image_path: str, **kwargs) -> Component:
    """Returns Component from a png image.

    Args:
        image_path: png file path.

    Keyword Args:
        nm_per_pixel: scale_factor.
        layer: layer tuple to output gds.
        threshold: value along which to find contours in the array.

    """
    import matplotlib.pyplot as plt

    # Load the image using matplotlib
    img = plt.imread(image_path)

    if len(img.shape) == 3:
        img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]

    # Convert image to numpy array (in fact, plt.imread already returns a numpy array)
    img_array = np.array(img)

    return from_np(img_array, **kwargs)


if __name__ == "__main__":
    from gdsfactory.config import PATH

    # import gdsfactory as gf
    # c1 = gf.components.straight()
    # c1 = gf.components.bend_circular()
    # c1 = gf.components.ring_single()
    # img = c1.to_np()
    # c2 = from_np(img)
    # c2.show()

    c = from_image(
        PATH.module / "samples" / "images" / "logo.png", nm_per_pixel=500, invert=True
    )
    c.show()
