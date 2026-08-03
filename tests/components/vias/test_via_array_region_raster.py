from __future__ import annotations

import gdsfactory as gf
from gdsfactory.components.vias.via_stack import (
    via_array_region_raster,
    via_array_stack_oa_compliant,
)


def test_via_array_region_raster_rectangular_region() -> None:
    gf.gpdk.PDK.activate()
    c = via_array_region_raster(
        region=[(0, 0), (5, 0), (5, 3), (0, 3)],
        bottom_layer="M1",
        via_layer="VIA1",
        top_layer="M2",
        via_x_minimum_cut_size=0.3,
        via_y_minimum_cut_size=0.3,
        via_x_minimum_enclosure=0.06,
        via_y_minimum_enclosure=0.06,
        via_x_minimum_spacing=0.3,
        via_y_minimum_spacing=0.3,
    )
    assert c.info["num_vias"] > 0
    assert c.info["bottom_layer"] == "M1"
    assert c.info["top_layer"] == "M2"
    assert c.info["via_x_cut_size"] == 0.3
    assert c.info["via_y_cut_size"] == 0.3


def test_via_array_region_raster_rectangle_via_type() -> None:
    gf.gpdk.PDK.activate()
    c = via_array_region_raster(
        region=[(0, 0), (5, 0), (5, 3), (0, 3)],
        bottom_layer="M1",
        via_layer="VIA1",
        top_layer="M2",
        via_type="rectangle",
        via_x_minimum_cut_size=0.3,
        via_y_minimum_cut_size=0.3,
        via_x_minimum_enclosure=0.06,
        via_y_minimum_enclosure=0.06,
        via_x_minimum_spacing=0.3,
        via_y_minimum_spacing=0.3,
    )
    assert c.info["num_vias"] > 0
    cut_w = c.info["via_x_cut_size"]
    cut_h = c.info["via_y_cut_size"]
    assert cut_w != cut_h, "Rectangle via type should double one dimension"


def test_via_array_stack_oa_compliant_size_input() -> None:
    gf.gpdk.PDK.activate()
    c = via_array_stack_oa_compliant(
        bottom_layer="M1",
        top_layer="M3",
        size=(5, 5),
        via_type="square",
    )
    assert c.info["bottom_layer"] == "M1"
    assert c.info["top_layer"] == "M3"
    assert c.info["num_via_layers"] == 2
    assert c.info["total_num_vias"] > 0
    assert len(c.info["per_layer_info"]) == 2


def test_via_array_stack_oa_compliant_region_input() -> None:
    gf.gpdk.PDK.activate()
    from shapely.geometry import Polygon as ShapelyPolygon

    bottom = ShapelyPolygon([(0, 0), (8, 0), (8, 5), (0, 5)])
    top = ShapelyPolygon([(2, -1), (10, 2), (8, 6), (0, 3)])
    region = list(bottom.intersection(top).exterior.coords)

    c = via_array_stack_oa_compliant(
        bottom_layer="M1",
        top_layer="M3",
        region=region,
        via_type="square",
    )
    assert c.info["total_num_vias"] > 0
    assert c.info["num_via_layers"] == 2
    for layer_info in c.info["per_layer_info"]:
        assert layer_info["num_vias"] > 0
        assert layer_info["via_x_cut_size"] > 0
