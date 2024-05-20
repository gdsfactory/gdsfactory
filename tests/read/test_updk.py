from gdsfactory.read.from_updk import from_updk
from gdsfactory.samples.pdk.fab_c import pdk


def test_updk() -> None:
    yaml_pdk_decription = pdk.to_updk()
    gdsfactory_script = from_updk(yaml_pdk_decription)
    assert gdsfactory_script


if __name__ == "__main__":
    test_updk()
