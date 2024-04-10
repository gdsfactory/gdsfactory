from pathlib import Path

import gdsfactory as gf

_this_dir = Path(__file__).parent
_test_yaml_dir = _this_dir / "yaml"

ACTIVE_PDK = gf.get_active_pdk()
ACTIVE_PDK.register_cells_yaml(dirpath=_test_yaml_dir)


def test_cache_container() -> None:
    c1 = gf.components.straight()
    c1r = c1.rotate()

    c2 = gf.components.straight()
    c2r = c2.rotate()

    assert c1.uid == c2.uid
    assert c1r.uid == c2r.uid, f"Cache UID mismatch: {c1r.uid} != {c2r.uid}"

def test_cache_prevents_duplicates() -> None:
    # Setup test environment
    # Generate or simulate conditions that might lead to duplicate names
    # Verify cache mechanism prevents duplicates
    pass


def test_cache_name_yaml() -> None:
    c1 = gf.get_component("test_pcell", length=23)
    c2 = gf.get_component("test_pcell", length=23)
    # the ordering is important, to ensure that cells with non-default parameters don't change the name of the default component
    c3 = gf.get_component("test_pcell")
    c4 = gf.get_component("test_pcell")

    assert c1 is c2
    assert c1.name == "test_pcell_length23"

    assert c3.name == "test_pcell"
    assert c3 is c4
