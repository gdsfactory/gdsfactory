from collections.abc import Iterator
from pathlib import Path

import kfactory as kf
import pytest

import gdsfactory as gf
from gdsfactory import partial
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
    assert xs.width == 1


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


def test_container_cell_conflict_raises_for_string_fast_path() -> None:
    """String lookups must still validate global cell/container name conflicts."""
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
        pdk.get_component("straight")


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
    gf.clear_cache()
    try:
        yield
    finally:
        gf.clear_cache()
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


def test_pdk_dbu_change_on_reactivation_after_cells_raises(
    restore_kcl_state: None,
) -> None:
    """Re-activating the active PDK cannot rescale a layout that holds cells.

    Switching to a *different* Pdk instance empties the layout first, so only a
    same-instance re-activation can reach this guard.
    """
    pdk = gf.Pdk(
        name="dbu_blocked",
        layers=LAYER,
        cross_sections={"strip": gf.cross_section.strip},
    )
    pdk.activate(force=True)
    gf.components.straight()
    assert len(gf.kcl.kcells) > 0

    pdk.dbu = 0.0005
    with pytest.raises(ValueError, match=r"cell\(s\) already exist"):
        pdk.activate(force=True)


def test_pdk_switch_clears_cells(restore_kcl_state: None) -> None:
    gf.components.straight()
    assert len(gf.kcl.kcells) > 0

    pdk = gf.Pdk(
        name="dbu_switch",
        layers=LAYER,
        cross_sections={"strip": gf.cross_section.strip},
        dbu=0.0005,
    )
    with pytest.warns(UserWarning, match="discards"):
        pdk.activate(force=True)

    assert gf.kcl.dbu == 0.0005
    assert len(gf.kcl.kcells) == 0


def test_pdk_switch_does_not_reserve_cells_by_name(restore_kcl_state: None) -> None:
    """Two PDKs can generate the same cell name for different geometry.

    Cell names serialize cross-sections and layers to their name, and with
    ``CONF.cell_layout_cache`` on, the decorator re-serves any existing cell with
    the computed name. Activating a PDK must therefore drop the previous PDK's
    cells, or the second factory silently returns the first one's geometry.
    """
    from gdsfactory.gpdk import PDK

    @gf.cell(basename="pdk_scoped_cell", register_factory=False)
    def narrow() -> gf.Component:
        c = gf.Component()
        c.add_polygon([(0, 0), (10, 0), (10, 1), (0, 1)], layer="WG")
        return c

    @gf.cell(basename="pdk_scoped_cell", register_factory=False)
    def wide() -> gf.Component:
        c = gf.Component()
        c.add_polygon([(0, 0), (20, 0), (20, 2), (0, 2)], layer="WG")
        return c

    PDK.activate(force=True)
    assert narrow().dbbox().height() == 1.0

    other = gf.Pdk(
        name="pdk_scoped_other",
        layers=LAYER,
        cross_sections={"strip": gf.cross_section.strip},
    )
    with pytest.warns(UserWarning, match="discards"):
        other.activate()

    assert wide().dbbox().height() == 2.0


def test_pdk_switch_rebuilds_vcell(restore_kcl_state: None) -> None:
    """A @vcell must not be re-served from the previous PDK after a switch.

    Virtual cells live outside the layout, so clear_kcells() cannot reach them and
    the decorator has no destroyed()-based self-healing. Since a cross_section
    serializes to its name, the two PDKs below share a single cache key, so only
    dropping the virtual factory caches can force the rebuild. They share a name
    too, which must not stop the switch from doing it.
    """
    narrow = gf.Pdk(
        name="vcell_cache",
        layers=LAYER,
        cross_sections={"strip": partial(gf.cross_section.strip, width=0.5)},
    )
    narrow.activate(force=True)
    assert gf.components.straight_all_angle(length=10).dbbox().height() == 0.5

    wide = gf.Pdk(
        name="vcell_cache",
        layers=LAYER,
        cross_sections={"strip": partial(gf.cross_section.strip, width=2.0)},
    )
    wide.activate(force=True)

    assert gf.components.straight_all_angle(length=10).dbbox().height() == 2.0


def test_pdk_same_dbu_with_existing_cells_allowed(restore_kcl_state: None) -> None:
    """Re-activating the active PDK with an unchanged DBU keeps the cells.

    Switching to a *different* Pdk instance clears the caches first, so only a
    same-instance re-activation can reach the guard with cells present.
    """
    pdk = gf.Pdk(
        name="dbu_same",
        layers=LAYER,
        cross_sections={"strip": gf.cross_section.strip},
        dbu=gf.kcl.dbu,
    )
    pdk.activate(force=True)
    gf.components.straight()
    assert len(gf.kcl.kcells) > 0

    pdk.activate(force=True)  # same PDK and DBU, so the cells are kept

    assert len(gf.kcl.kcells) > 0


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
    """A layer outside the PDK that holds shapes is kept, not pruned (#4595).

    Pruning must never orphan geometry. Switching to a *different* Pdk instance
    empties the layout first, so the case only arises on a same-instance
    re-activation.
    """

    class MyFabLayers(LayerMap):
        MY_WG = (10, 0)

    pdk = gf.Pdk(
        name="keep_fab",
        layers=MyFabLayers,
        cross_sections={"strip": gf.cross_section.strip},
    )
    pdk.activate(force=True)

    c = gf.Component()
    c.add_polygon([(0, 0), (5, 0), (5, 5), (0, 5)], layer=(1, 0))  # foreign layer
    assert (1, 0) in _registered_layers()

    pdk.activate(force=True)  # same PDK, so the layout is not cleared

    assert (1, 0) in _registered_layers()  # kept because it still holds shapes
    assert not c.shapes(gf.kcl.layout.find_layer(1, 0)).is_empty()


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


def test_reactivate_pdk_moves_wrong_index_layer_with_geometry(
    restore_kcl_state: None,
) -> None:
    """Re-activating a PDK moves wrongly registered geometry back to enum indexes.

    Registering WG away from its LayerEnum index is what happens after building
    under a PDK that does not declare it; here it is set up directly, because
    switching PDK now empties the layout before the layers are restored.
    """
    gf.gpdk.PDK.activate(force=True)

    layout = gf.kcl.layout
    info = kf.kdb.LayerInfo(1, 0)
    layout.delete_layer(layout.find_layer(info))
    wrong_index = max(layout.layer_indexes()) + 1  # free, and not WG's index
    assert not layout.is_valid_layer(wrong_index)
    layout.insert_layer_at(wrong_index, info)
    assert layout.find_layer(info) == wrong_index != int(LAYER.WG)

    c = gf.Component()
    c.add_polygon([(0, 0), (5, 0), (5, 5), (0, 5)], layer=(1, 0))
    assert not c.shapes(wrong_index).is_empty()

    gf.gpdk.PDK.activate(force=True)  # same PDK, so the layout is not cleared

    assert layout.find_layer(info) == int(LAYER.WG)
    assert not c.shapes(int(LAYER.WG)).is_empty()


def test_get_layer_name_exception_chaining() -> None:
    pdk = _make_pdk()
    with pytest.raises(ValueError) as exc_info:
        pdk.get_layer_name((999, 999))

    assert "Could not find name for layer" in str(exc_info.value)
    # Ensure that exception chaining has occurred with the inner exception, which should be ValueError
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, (ValueError, KeyError, TypeError))


def test_get_cross_section_instance_applies_kwargs() -> None:
    """Overrides apply to CrossSection instances like they do for string specs (#4588)."""
    xs = gf.get_cross_section("strip")
    xs_wide = gf.get_cross_section(xs, width=2)
    assert xs_wide.width == 2
    # the copy gets a derived name so it caches separately from the original
    assert xs_wide.name != xs.name
    # no overrides still returns the instance unchanged
    assert gf.get_cross_section(xs) is xs


def test_get_cross_section_dict_applies_kwargs() -> None:
    """Overrides win over the dict spec's own settings."""
    spec = {"cross_section": "strip", "settings": {"width": 1}}
    assert gf.get_cross_section(spec).width == 1
    assert gf.get_cross_section(spec, width=3.0).width == 3.0
    with pytest.raises(kf.exceptions.CrossSectionNamingConflictError):
        gf.get_cross_section(spec, radius=20)
    # the override does not write back into spec["settings"]
    assert spec == {"cross_section": "strip", "settings": {"width": 1}}


def test_get_cross_section_kfactory_applies_kwargs() -> None:
    """A kfactory cross_section takes overrides like every other spec does."""
    kf_xs = gf.components.straight().ports[0].cross_section
    registered = gf.get_cross_section(kf_xs)

    # the kfactory name describes the original, so an override cannot keep it
    assert gf.get_cross_section(kf_xs, width=2.0).width == 2.0
    assert gf.get_cross_section(kf_xs, width=2.0).name != registered.name

    # Radius belongs to the bend/route call, not a profile override.
    with pytest.raises(ValueError, match="Only width"):
        gf.get_cross_section(kf_xs, radius=registered.radius)


def test_get_cross_section_kfactory_unknown_layer_keeps_index() -> None:
    """A profile on an unnameable layer keeps its physical LayerInfo."""
    enclosure = kf.LayerEnclosure(main_layer=kf.kdb.LayerInfo(999, 999), kcl=gf.kcl)
    sxs = kf.SymmetricalCrossSection(
        width=1000, enclosure=enclosure, name="unnamed_layer_xs"
    )

    xs = gf.get_cross_section(sxs)
    assert xs.name == "unnamed_layer_xs"
    assert xs.width == 1.0
    assert xs.layer == kf.kdb.LayerInfo(999, 999)


def test_get_cross_section_invalid_spec_raises() -> None:
    with pytest.raises(ValueError, match="cross_section name is required"):
        gf.get_cross_section({"settings": {"width": 1}})

    with pytest.raises(ValueError, match="expects a CrossSectionSpec"):
        gf.get_cross_section(42)  # type: ignore[arg-type]


def test_taper_cross_section_instance_matches_str_spec() -> None:
    """Taper with a resolved CrossSection tapers like the string spec (#4588).

    The instance form is built first: it must not depend on a previously
    cached string-spec cell.
    """
    t_obj = gf.components.taper(width2=10, cross_section=gf.get_cross_section("strip"))
    t_str = gf.components.taper(width2=10, cross_section="strip")
    assert [p.width for p in t_obj.ports] == [0.5, 10.0]
    assert [p.width for p in t_str.ports] == [0.5, 10.0]


def test_get_cell_dict_applies_kwargs() -> None:
    """Overrides win over the dict spec's own settings."""
    spec = {"function": "straight", "settings": {"length": 5}}
    assert gf.get_cell(spec)().settings.length == 5
    assert gf.get_cell(spec, length=20)().settings.length == 20
    # the override does not write back into spec["settings"]
    assert spec == {"function": "straight", "settings": {"length": 5}}

    with pytest.raises(ValueError, match="Invalid setting 'bogus'"):
        gf.get_cell({"function": "straight", "bogus": 1})

    # get_cell only reads "function", so a "component" key resolves to nothing
    with pytest.raises(ValueError, match="not in cells"):
        gf.get_cell({"component": "straight"})


def test_get_component_dict_applies_kwargs() -> None:
    """Kwargs beat the dict spec's settings, and the settings argument beats kwargs."""
    spec = {"component": "straight", "settings": {"length": 5}}
    assert gf.get_component(spec).settings.length == 5
    assert gf.get_component(spec, length=20).settings.length == 20
    assert (
        gf.get_component(spec, settings={"length": 7}, length=20).settings.length == 7
    )
    # the override does not write back into spec["settings"]
    assert spec == {"component": "straight", "settings": {"length": 5}}

    # a serialized "function" keeps only the last part of the dotted path
    dotted = {"function": "gdsfactory.components.straight", "settings": {"length": 3}}
    assert gf.get_component(dotted, length=9).settings.length == 9

    with pytest.raises(ValueError, match="Invalid setting 'bogus'"):
        gf.get_component({"component": "straight", "bogus": 1})

    # neither "component" nor "function" is given
    with pytest.raises(ValueError, match=r"'None'.*not in cells"):
        gf.get_component({"settings": {"length": 1}})
