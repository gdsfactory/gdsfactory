from gdsfactory.samples.pdk.fab_c import pdk
from gdsfactory.read.from_updk import from_updk


def test_updk() -> None:
    yaml_pdk_decription = pdk.to_updk()
    gdsfactory_script = from_updk(yaml_pdk_decription)
    assert gdsfactory_script
