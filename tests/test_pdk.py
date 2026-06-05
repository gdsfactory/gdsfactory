from collections.abc import Iterator

import pytest

import gdsfactory as gf
from gdsfactory.gpdk import LAYER


def test_get_cross_section() -> None:
    assert gf.pdk.get_cross_section("strip") == gf.cross_section.strip()
    assert (
        gf.pdk.get_cross_section(gf.cross_section.strip()) == gf.cross_section.strip()
    )
    cross_section = {"cross_section": "strip", "settings": {"width": 1}}
    xs = gf.get_cross_section(cross_section)
    assert xs.sections[0].width == 1


def test_get_layer() -> None:
    assert gf.get_layer(1) == LAYER.WG
    assert gf.get_layer((1, 0)) == LAYER.WG
    assert gf.get_layer("WG") == LAYER.WG


def test_container_cell_conflict_raises_error() -> None:
    """Test that a cell with the same name as a container raises an error."""
    pdk = gf.Pdk(
        name="test",
        layers=LAYER,
        cross_sections={"strip": gf.cross_section.strip},
        cells={
            "straight": gf.components.straight,
            "add_pads_top": gf.containers.add_pads_top,
        },
        containers={"add_pads_top": gf.containers.add_pads_top},
    )

    with pytest.raises(ValueError, match=r".* overlapping cell names .*add_pads_top.*"):
        pdk.get_component("add_pads_top")


def _make_pdk() -> gf.Pdk:
    return gf.Pdk(
        name="test",
        layers=LAYER,
        cross_sections={"strip": gf.cross_section.strip},
    )


def test_pdk_has_pydantic_slots() -> None:
    """Pdk.__init__ must initialise every BaseModel slot it overrides.

    Regression guard for #4485: a missing slot (e.g. __pydantic_extra__) only
    surfaces when downstream code touches it, so assert each slot directly.
    """
    pdk = _make_pdk()
    # All slots Pdk.__init__ writes via object.__setattr__ must be present.
    assert isinstance(pdk.__dict__, dict)
    assert isinstance(pdk.__pydantic_fields_set__, set)
    assert pdk.__pydantic_private__ is not None
    # __pydantic_extra__ is the slot that broke in 9.40.0; reading it must not raise.
    assert pdk.__pydantic_extra__ is None or isinstance(pdk.__pydantic_extra__, dict)


def test_pdk_copy() -> None:
    """Regression test for #4485: copy.copy(pdk) must not raise AttributeError."""
    import copy

    pdk = _make_pdk()
    pdk_copy = copy.copy(pdk)
    assert pdk_copy.name == pdk.name
    assert pdk_copy.cross_sections == pdk.cross_sections


def test_pdk_deepcopy() -> None:
    """Deepcopy uses the same slot machinery as copy and must also work."""
    import copy

    pdk = _make_pdk()
    pdk_copy = copy.deepcopy(pdk)
    assert pdk_copy.name == pdk.name
    assert pdk_copy.cross_sections.keys() == pdk.cross_sections.keys()


def test_pdk_pickle_roundtrip() -> None:
    """Pickling exercises __getstate__/__setstate__ which depend on the slots."""
    import pickle

    pdk = _make_pdk()
    restored = pickle.loads(pickle.dumps(pdk))
    assert restored.name == pdk.name
    assert restored.cross_sections.keys() == pdk.cross_sections.keys()


def test_pdk_model_dump() -> None:
    """model_dump walks __pydantic_fields_set__/__pydantic_extra__ — guard it."""
    pdk = _make_pdk()
    dumped = pdk.model_dump()
    assert dumped["name"] == "test"


@pytest.fixture
def restore_kcl_state() -> Iterator[None]:
    from gdsfactory.gpdk import PDK

    original_dbu = gf.kcl.dbu
    gf.kcl.clear_kcells()
    try:
        yield
    finally:
        gf.kcl.clear_kcells()
        gf.kcl.dbu = original_dbu
        PDK.activate(force=True)


def test_pdk_sets_dbu(restore_kcl_state: None) -> None:
    pdk = gf.Pdk(
        name="dbu_test",
        layers=LAYER,
        cross_sections={"strip": gf.cross_section.strip},
        dbu=0.0005,
    )
    pdk.activate(force=True)
    assert gf.kcl.dbu == 0.0005


def test_pdk_dbu_change_after_cells_raises(restore_kcl_state: None) -> None:
    gf.components.straight()
    assert len(gf.kcl.kcells) > 0

    pdk = gf.Pdk(
        name="dbu_blocked",
        layers=LAYER,
        cross_sections={"strip": gf.cross_section.strip},
        dbu=0.0005,
    )
    with pytest.raises(ValueError, match=r"cell\(s\) already exist"):
        pdk.activate(force=True)


def test_pdk_same_dbu_with_existing_cells_allowed(restore_kcl_state: None) -> None:
    gf.components.straight()
    assert len(gf.kcl.kcells) > 0

    existing_dbu = gf.kcl.dbu

    pdk = gf.Pdk(
        name="dbu_same",
        layers=LAYER,
        cross_sections={"strip": gf.cross_section.strip},
        dbu=existing_dbu,
    )
    pdk.activate(force=True)


def test_get_layer_name_exception_chaining() -> None:
    pdk = _make_pdk()
    with pytest.raises(ValueError) as exc_info:
        pdk.get_layer_name((999, 999))

    assert "Could not find name for layer" in str(exc_info.value)
    # Ensure that exception chaining has occurred with the inner exception, which should be ValueError
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, (ValueError, KeyError, TypeError))
