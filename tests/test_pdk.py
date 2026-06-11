from collections.abc import Iterator
from pathlib import Path

import pytest

import gdsfactory as gf
from gdsfactory.config import CONF
from gdsfactory.gpdk import LAYER
from gdsfactory.technology import LayerMap


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


def _registered_layers() -> set[tuple[int, int]]:
    layout = gf.kcl.layout
    return {
        (layout.get_info(i).layer, layout.get_info(i).datatype)
        for i in layout.layer_indexes()
    }


def test_activate_custom_pdk_prunes_generic_layers(
    restore_kcl_state: None, tmp_path: Path
) -> None:
    """Activating a custom PDK drops the import-time generic layermap (#4595).

    The generic layers are then no longer registered, written out, or shown.
    """

    class MyFabLayers(LayerMap):
        MY_WG = (10, 0)
        MY_SLAB = (11, 0)

    pdk = gf.Pdk(
        name="prune_fab",
        layers=MyFabLayers,
        cross_sections={"strip": gf.cross_section.strip},
    )
    pdk.activate(force=True)

    registered = _registered_layers()
    assert (1, 0) not in registered  # generic WG no longer registered
    assert {(10, 0), (11, 0)} <= registered  # the active PDK's layers remain

    c = gf.Component()
    c.add_polygon([(0, 0), (10, 0), (10, 5), (0, 5)], layer=(10, 0))
    path = tmp_path / "prune_fab.oas"
    c.write(path)

    import klayout.db as kdb

    layout = kdb.Layout()
    layout.read(str(path))
    written = {
        (layout.get_info(i).layer, layout.get_info(i).datatype)
        for i in layout.layer_indexes()
    }
    assert (10, 0) in written  # the active layer is written
    assert (1, 0) not in written  # generic layers are not


def test_activate_custom_pdk_keeps_layers_with_geometry(
    restore_kcl_state: None,
) -> None:
    """A layer outside the new PDK that holds shapes is kept, not pruned (#4595).

    Switching PDKs must never silently discard geometry.
    """
    gf.gpdk.PDK.activate(force=True)
    c = gf.Component()
    c.add_polygon([(0, 0), (5, 0), (5, 5), (0, 5)], layer=(1, 0))  # WG holds geometry

    class MyFabLayers(LayerMap):
        MY_WG = (10, 0)

    pdk = gf.Pdk(
        name="keep_fab",
        layers=MyFabLayers,
        cross_sections={"strip": gf.cross_section.strip},
    )
    pdk.activate(force=True)

    assert (1, 0) in _registered_layers()  # kept because it still holds shapes


def test_activate_custom_pdk_preserves_error_layer(
    restore_kcl_state: None,
) -> None:
    """The on-demand routing-error layer is never pruned (#4595).

    Even with no geometry on it, ``CONF.layer_error_path`` is kept when a custom
    PDK that omits it is activated, so routing-error markers still have a home.
    """
    import klayout.db as kdb

    gf.gpdk.PDK.activate(force=True)
    error_layer = tuple(CONF.layer_error_path)
    # Register the error layer slot with no shapes, so only the explicit keep --
    # not the holds-geometry fallback -- can save it from pruning.
    gf.kcl.layout.layer(kdb.LayerInfo(error_layer[0], error_layer[1]))
    assert error_layer in _registered_layers()

    class MyFabLayers(LayerMap):
        MY_WG = (10, 0)

    pdk = gf.Pdk(
        name="error_layer_fab",
        layers=MyFabLayers,
        cross_sections={"strip": gf.cross_section.strip},
    )
    pdk.activate(force=True)

    assert error_layer in _registered_layers()  # error layer is always kept
    assert (1, 0) not in _registered_layers()  # but generic layers still pruned


def test_get_layer_name_exception_chaining() -> None:
    pdk = _make_pdk()
    with pytest.raises(ValueError) as exc_info:
        pdk.get_layer_name((999, 999))

    assert "Could not find name for layer" in str(exc_info.value)
    # Ensure that exception chaining has occurred with the inner exception, which should be ValueError
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, (ValueError, KeyError, TypeError))
