from gdsfactory.config import GDSDIR_TEMP
from gdsfactory.generic_tech import get_generic_pdk
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
]


def test_updk() -> None:
    PDK = get_generic_pdk()
    yaml_pdk_description = PDK.to_updk(exclude=exclude)
    filepath = GDSDIR_TEMP / "pdk.yaml"
    GDSDIR_TEMP.mkdir(exist_ok=True)
    filepath.write_text(yaml_pdk_description)
    gdsfactory_script = from_updk(filepath)
    assert gdsfactory_script


if __name__ == "__main__":
    # test_updk()
    PDK = get_generic_pdk()
    yaml_pdk_description = PDK.to_updk(exclude=exclude)
    filepath = GDSDIR_TEMP / "pdk.yaml"
    GDSDIR_TEMP.mkdir(exist_ok=True)
    filepath.write_text(yaml_pdk_description)
    gdsfactory_script = from_updk(filepath)
