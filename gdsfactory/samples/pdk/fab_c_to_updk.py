"""Write FabC example PDK into uPDK YAML format."""

from __future__ import annotations
from gdsfactory.samples.pdk.fab_c import pdk


if __name__ == "__main__":
    yaml_pdk_decription = pdk.to_updk()
    print(yaml_pdk_decription)
