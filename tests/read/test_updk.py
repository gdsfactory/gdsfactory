from gdsfactory.config import GDSDIR_TEMP
from gdsfactory.generic_tech import get_generic_pdk
from gdsfactory.read.from_updk import from_updk


def test_updk() -> None:
    from gdsfactory.samples.pdk.fab_c import PDK

    PDK.activate()
    yaml_pdk_decription = PDK.to_updk()
    filepath = GDSDIR_TEMP / "pdk.yaml"
    filepath.write_text(yaml_pdk_decription)
    gdsfactory_script = from_updk(filepath)
    assert gdsfactory_script

    PDK = get_generic_pdk()
    PDK.activate()
