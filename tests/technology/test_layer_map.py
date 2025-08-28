import pathlib

import pytest

from gdsfactory.technology.layer_map import lyp_to_dataclass


def test_lyp_to_dataclass(tmp_path: pathlib.Path) -> None:
    """Test converting KLayout lyp file to LayerMap dataclass."""
    lyp_content = """<?xml version="1.0" encoding="utf-8"?>
<layer-properties>
 <properties>
  <frame-color>#ff0000</frame-color>
  <fill-color>#ff0000</fill-color>
  <frame-brightness>0</frame-brightness>
  <fill-brightness>0</fill-brightness>
  <dither-pattern>1</dither-pattern>
  <line-style/>
  <valid>true</valid>
  <visible>true</visible>
  <transparent>false</transparent>
  <width>1</width>
  <marked>false</marked>
  <xfill>false</xfill>
  <animation>0</animation>
  <name>M1</name>
  <source>1/0@1</source>
 </properties>
</layer-properties>"""

    lyp_file = tmp_path / "test.lyp"
    lyp_file.write_text(lyp_content)

    script = lyp_to_dataclass(lyp_file)
    assert "M1: Layer = (1, 0)" in script
    assert "class LayerMapFab(LayerMap):" in script

    py_file = lyp_file.with_suffix(".py")
    py_file.write_text("existing content")
    with pytest.raises(FileExistsError):
        lyp_to_dataclass(lyp_file, overwrite=False)

    with pytest.raises(FileNotFoundError):
        lyp_to_dataclass("nonexistent.lyp")
