import re
from pathlib import Path

import gdsfactory as gf
from gdsfactory.component import GDSDIR_TEMP
from gdsfactory.export.to_svg import (
    to_svg,
)
from gdsfactory.typings import Layer


def test_to_svg() -> None:
    """Test the to_svg function to ensure it correctly generates an SVG file from a Component."""
    # Step 1: Create a simple component (rectangle) on a known layer
    test_layer: Layer = (1, 0)  # Example layer (10, 0)
    component = gf.c.grating_coupler_elliptical_trenches()

    # Step 2: Define the SVG filename within the temporary directory
    svg_filename: Path = GDSDIR_TEMP / "test_component.svg"

    # Step 3: Call the to_svg function
    to_svg(
        component=component,
        exclude_layers=None,  # No layers excluded
        filename=str(svg_filename),
        scale=1.0,  # No scaling
    )

    # Step 4: Verify that the SVG file was created
    assert svg_filename.exists(), "SVG file was not created."

    # Step 5: Verify that the SVG file is not empty
    assert svg_filename.stat().st_size > 0, "SVG file is empty."

    # Step 6: Read the SVG content
    with open(svg_filename) as f:
        svg_content = f.read()

    # Step 7: Verify SVG header
    assert (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>' in svg_content
    ), "SVG header is missing."

    # Step 8: Verify SVG root element
    assert "<svg" in svg_content, "SVG does not contain <svg> tag."
    assert "</svg>" in svg_content, "SVG does not contain closing </svg> tag."

    # Step 9: Verify layer group
    layer_id = f'id="layer{test_layer[0]:03d}_datatype{test_layer[1]:03d}"'
    assert (
        layer_id in svg_content
    ), f"SVG does not contain expected layer group {layer_id}."

    # Step 10: Verify path element within the layer group
    assert "<path" in svg_content, "SVG does not contain any <path> elements."
    assert 'd="' in svg_content, "SVG <path> element does not contain 'd' attribute."

    # Step 11: Optionally, verify specific path data (coordinates)
    # Extract the path data
    path_match = re.search(r'd="([^"]+)"', svg_content)
    assert path_match is not None, "SVG <path> element does not contain 'd' attribute."
