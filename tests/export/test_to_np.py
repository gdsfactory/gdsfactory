from typing import Literal

import numpy as np
import pytest

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components import bend_circular, straight
from gdsfactory.export.to_np import to_np
from gdsfactory.typings import LayerSpecs


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


def test_to_np_with_layers_none() -> None:
    c = gf.Component()
    c << gf.components.rectangle(size=(1, 1), layer=(1, 0))
    c << gf.components.rectangle(size=(1, 1), layer=(2, 0))

    img = to_np(c, nm_per_pixel=100, layers=None)

    assert np.max(img) == 1


def test_to_np_forwards_layer_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    c = straight()
    layers = ((1, 0),)
    original_get_polygons_points = Component.get_polygons_points
    called_with: dict[str, LayerSpecs | None] = {}

    def spy_get_polygons_points(
        self: Component,
        merge: bool = False,
        scale: float | None = None,
        by: Literal["index", "name", "tuple"] = "index",
        layers: LayerSpecs | None = None,
    ) -> dict[object, object]:
        called_with["layers"] = layers
        return original_get_polygons_points(
            self, merge=merge, scale=scale, by=by, layers=layers
        )

    monkeypatch.setattr(Component, "get_polygons_points", spy_get_polygons_points)

    to_np(c, nm_per_pixel=20, layers=layers)

    assert called_with["layers"] == layers


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
