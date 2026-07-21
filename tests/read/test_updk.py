from pathlib import Path

import pytest

from gdsfactory.config import GDSDIR_TEMP
from gdsfactory.gpdk import get_generic_pdk
from gdsfactory.read.from_updk import from_updk

exclude = [
    "add_fiber_array_optical_south_electrical_north",
    "bbox",
    "component_sequence",
    "extend_ports_list",
    "pack_doe",
    "pack_doe_grid",
    "straight_piecewise",
    "text_freetype",
    "ring_double_pn",
    "grating_coupler_elliptical_lumerical_etch70",
]


def test_updk_generic() -> None:
    PDK = get_generic_pdk()
    yaml_pdk_description = PDK.to_updk(exclude=exclude)
    filepath = GDSDIR_TEMP / "pdk.yaml"
    GDSDIR_TEMP.mkdir(exist_ok=True)
    filepath.write_text(yaml_pdk_description)
    gdsfactory_script = from_updk(filepath)
    assert gdsfactory_script


def test_updk_custom_parameter_format(tmp_path: Path) -> None:
    filepath = tmp_path / "pdk.yaml"
    filepath.write_text(
        """
blocks:
  phase_shifter:
    parameters:
      length:
        type: float
        value: 10
        doc: Length
    bbox:
      - [0, 0]
      - [10, 0]
      - [10, 1]
      - [0, 1]
"""
    )

    script = from_updk(filepath, parameter_format="{name}:{value}")

    assert "name = f'phase_shifter:length:{length}'" in script


def test_updk_rejects_unknown_parameter_format_placeholder(tmp_path: Path) -> None:
    filepath = tmp_path / "pdk.yaml"
    filepath.write_text(
        """
blocks:
  straight:
    parameters:
      length:
        type: float
        value: 10
        doc: Length
    bbox: [[0, 0], [10, 0], [10, 1], [0, 1]]
"""
    )

    with pytest.raises(ValueError, match="supports only"):
        from_updk(filepath, parameter_format="{unknown}")
