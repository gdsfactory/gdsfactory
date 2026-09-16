"""Read component from a numpy.ndarray."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

import gdsfactory as gf
from gdsfactory.component import (
    Component,
    boolean_not,
    boolean_or,
    points_to_polygon,
)
from gdsfactory.typings import PathType


def compute_area_signed(pr: npt.NDArray[np.floating[Any]]) -> float:
    """Return the signed area enclosed by a ring using the linear time.

    algorithm at http://www.cgafaq.info/wiki/Polygon_Area. A value >= 0
    indicates a counter-clockwise oriented ring.

    """
    xs, ys = map(list, zip(*pr, strict=False))
    xs.append(xs[1])
    ys.append(ys[1])
    xs_ = cast("list[float]", xs)
    ys_ = cast("list[float]", ys)
    return sum(xs_[i] * (ys_[i + 1] - ys_[i - 1]) for i in range(1, len(pr))) / 2.0


def from_np(
    ndarray: npt.NDArray[np.floating[Any]],
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

    ndarray = np.pad(ndarray, 2)
    contours = measure.find_contours(ndarray, threshold)
    assert len(contours) > 0, (
        f"no contours found for threshold = {threshold}, maybe you can reduce the"
        " threshold"
    )

    if not invert:
        d = Component()
        for contour in contours:
            if compute_area_signed(contour) >= 0:
                d.add_polygon(contour * 1e-3 * nm_per_pixel, layer=layer)
        return d

    # Apply contours from the outside in so that islands nested inside holes are
    # restored after their enclosing hole is subtracted. Combining all positive
    # and negative contours separately loses this even-odd nesting information.
    c = Component()
    layer_index = gf.get_layer(layer)
    dbu = c.kcl.dbu
    region = gf.kdb.Region()
    for area, contour in sorted(
        ((compute_area_signed(contour), contour) for contour in contours),
        key=lambda item: abs(item[0]),
        reverse=True,
    ):
        poly = cast("gf.kdb.DPolygon", points_to_polygon(contour * 1e-3 * nm_per_pixel))
        contour_region = gf.kdb.Region(poly.to_itype(dbu))
        region = (
            boolean_or(region, contour_region)
            if area < 0
            else boolean_not(region, contour_region)
        )
    c.shapes(layer_index).insert(region)
    return c


@gf.cell
def from_image(image_path: PathType, **kwargs: Any) -> Component:
    """Returns Component from a png image.

    Args:
        image_path: png file path.
        kwargs: for from_np.

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
