import pathlib
import xml.etree.ElementTree as ET

import aenum

from gdsfactory.technology import LayerViews
from gdsfactory.technology.klayout_tech import KLayoutTechnology
from gdsfactory.technology.layer_views import LayerView


def test_klayout_tech_basic() -> None:
    """Test basic KLayoutTechnology functionality."""
    tech = KLayoutTechnology(
        name="test_tech", layer_map={"M1": (1, 0), "M2": (2, 0), "VIA1": (3, 0)}
    )
    assert tech.name == "test_tech"
    assert tech.layer_map["M1"] == (1, 0)


def test_klayout_tech_write(tmp_path: pathlib.Path) -> None:
    """Test writing technology files."""
    tech = KLayoutTechnology(
        name="test_tech",
        layer_map={"M1": (1, 0), "M2": (2, 0), "VIA1": (3, 0)},
        connectivity=[("M1", "VIA1", "M2")],
    )

    tech.write_tech(tech_dir=tmp_path)

    # Verify files were created
    assert (tmp_path / "tech.lyt").exists()
    assert (tmp_path / "d25").exists()

    # Test loading written tech file
    tech_xml = (tmp_path / "tech.lyt").read_bytes()
    root = ET.XML(tech_xml)

    # Verify connectivity was written correctly
    connections = root.findall(".//connection")
    assert len(connections) == 1
    assert connections[0].text == "M1,VIA1,M2"

    symbols = root.findall(".//symbols")
    assert len(symbols) == 2  # VIA1 and M2
    assert any(s.text is not None and "M2='2/0'" in s.text for s in symbols)
    assert any(s.text is not None and "VIA1='3/0'" in s.text for s in symbols)


def test_klayout_tech_mebes_config(tmp_path: pathlib.Path) -> None:
    """Test mebes configuration."""
    custom_mebes = {"invert": True, "data-layer": 5, "boundary-name": "TEST"}

    tech = KLayoutTechnology(name="test_tech", layer_map={"M1": (1, 0)})

    tech.write_tech(tech_dir=tmp_path, mebes_config=custom_mebes)

    tech_xml = (tmp_path / "tech.lyt").read_bytes()
    root = ET.XML(tech_xml)

    mebes = root.find(".//mebes")
    assert mebes is not None
    invert = mebes.find("invert")
    assert invert is not None
    assert invert.text == "true"
    data_layer = mebes.find("data-layer")
    assert data_layer is not None
    assert data_layer.text == "5"
    boundary_name = mebes.find("boundary-name")
    assert boundary_name is not None
    assert boundary_name.text == "TEST"


def test_klayout_tech_layer_views(tmp_path: pathlib.Path) -> None:
    """Test layer views integration."""
    layer_views = LayerViews(
        filepath=None, layer_views={"M1": LayerView(name="M1", fill_color="#FF0000")}
    )

    tech = KLayoutTechnology(
        name="test_tech",
        layer_map={"M1": (1, 0), "M2": (2, 0)},
        layer_views=layer_views,
    )

    tech.write_tech(tech_dir=tmp_path)
    assert (tmp_path / "layers.lyp").exists()


def test_klayout_tech_enum_layer_map() -> None:
    """Test using enum for layer map."""

    class TestLayers(aenum.Enum):  # type: ignore[misc]
        M1 = (1, 0)
        M2 = (2, 0)

        def __new__(cls, layer: int, datatype: int) -> "TestLayers":
            obj = object.__new__(cls)
            obj._value_ = (layer, datatype)
            obj.layer = layer
            obj.datatype = datatype
            return obj

    tech = KLayoutTechnology(name="test_tech", layer_map=TestLayers)  # type: ignore[arg-type]

    assert tech.layer_map["M1"] == (1, 0)
    assert tech.layer_map["M2"] == (2, 0)
