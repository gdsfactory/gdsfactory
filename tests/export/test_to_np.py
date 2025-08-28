import numpy as np

from gdsfactory.components import bend_circular, straight
from gdsfactory.export.to_np import to_np


def test_to_np_basic() -> None:
    c = straight()
    img = to_np(c, nm_per_pixel=20)
    assert img is not None
    assert img.shape[0] > 0 and img.shape[1] > 0


def test_to_np_different_nm_per_pixel() -> None:
    c = straight()
    img1 = to_np(c, nm_per_pixel=20)
    img2 = to_np(c, nm_per_pixel=10)
    assert img1.shape != img2.shape


def test_to_np_with_layers() -> None:
    c = straight()
    img = to_np(c, nm_per_pixel=20, layers=((1, 0), (2, 0)))
    assert np.max(img) == 1


def test_to_np_with_values() -> None:
    c = straight()
    img = to_np(c, nm_per_pixel=20, values=[0.5])
    assert np.max(img) == 0.5


def test_to_np_with_pad_width() -> None:
    c = straight()
    img = to_np(c, nm_per_pixel=20, pad_width=5)
    assert img.shape[0] > 0 and img.shape[1] > 0


def test_to_np_with_bend_circular() -> None:
    c = bend_circular()
    img = to_np(c, nm_per_pixel=20)
    assert img is not None
