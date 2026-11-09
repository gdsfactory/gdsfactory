"""Generic PDK material coverage, provenance and legacy numerical compatibility."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from pdk_schema import Index, MaterialCard

import gdsfactory as gf
from gdsfactory.gpdk.material_cards import (
    MATERIAL_ALIASES,
    MATERIAL_CARDS,
    get_refractive_index,
)
from gdsfactory.gpdk.simulation_settings import materials_index, si, sin, sio2


def test_every_stack_and_process_material_has_a_canonical_card() -> None:
    from gdsfactory.gpdk.layer_stack import get_layer_stack, get_process

    names = {level.material for level in get_layer_stack().layers.values()}
    assert names == {"Si", "SiO2", "SiN", "Ge", "Air", "Aluminum", "TiN"}
    assert names == MATERIAL_CARDS.keys()
    for step in get_process():
        name = getattr(step, "material", None)
        if name is not None:
            assert name in names
    for name, card in MATERIAL_CARDS.items():
        assert isinstance(card, MaterialCard)
        assert card.name == name
        assert MaterialCard.model_validate_json(card.model_dump_json()) == card


@pytest.mark.parametrize("name", list(MATERIAL_ALIASES))
def test_legacy_material_names_resolve_to_the_canonical_card(name: str) -> None:
    pdk = gf.gpdk.PDK
    canonical = MATERIAL_ALIASES[name]
    assert pdk.get_material_card(name) is pdk.material_cards[canonical]
    assert materials_index[name] is materials_index[canonical]


@pytest.mark.parametrize(("name", "legacy"), [("Si", si), ("SiO2", sio2), ("SiN", sin)])
def test_existing_material_indices_are_preserved(name: str, legacy: object) -> None:
    wavelengths = np.array([0.6, 1.0, 1.31, 1.55, 2.0])
    expected = legacy(wavelengths)  # type: ignore[operator]
    actual = get_refractive_index(name, wavelengths)
    np.testing.assert_allclose(actual.real, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual.imag, np.zeros(wavelengths.shape))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Ge", 4.25 + 0.02519j),
        ("Aluminum", 1.5785092140921408 + 15.658109304426379j),
        ("TiN", 2.171641930835735 + 8.003816829971182j),
        ("Air", 1 + 0j),
    ],
)
def test_published_reference_indices_at_1550_nm(name: str, expected: complex) -> None:
    """Check independent anchors from the cited reference tables, including loss."""
    assert get_refractive_index(name, 1.55) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("name", ["Si", "SiO2", "SiN", "Ge", "Aluminum", "TiN"])
def test_material_tables_refuse_extrapolation(name: str) -> None:
    card = MATERIAL_CARDS[name]
    assert card.optical is not None
    model = card.optical.permittivity
    assert isinstance(model, Index)
    assert model.validity is not None and model.validity.over is not None
    band = model.validity.over["wavelength"]
    for wavelength in (band.min / 2, band.max * 2):
        with pytest.raises(ValueError, match="wavelength"):
            get_refractive_index(name, wavelength)


def test_material_cards_and_aliases_are_inherited_and_can_be_overridden() -> None:
    base = gf.Pdk(
        name="material-base",
        material_cards={"Air": MATERIAL_CARDS["Air"]},
        material_aliases={"air": "Air"},
    )
    modified = MATERIAL_CARDS["Air"].model_copy(
        update={"info": {"purpose": "override"}}
    )
    derived = gf.Pdk(
        name="material-derived", base_pdks=[base], material_cards={"Air": modified}
    )

    assert derived.get_material_card("air") is modified
    assert base.get_material_card("air") is MATERIAL_CARDS["Air"]
    restored = gf.Pdk(
        name="material-restored",
        material_cards={"Air": modified.model_dump(mode="json")},
        material_aliases={"air": "Air"},
    )
    assert restored.get_material_card("air") == modified


def test_material_alias_cycles_and_missing_targets_are_rejected() -> None:
    with pytest.raises(ValueError, match="cycle"):
        gf.Pdk(name="cycle", material_aliases={"a": "b", "b": "a"})
    with pytest.raises(ValueError, match="missing"):
        gf.Pdk(name="missing", material_aliases={"a": "missing"})


@pytest.mark.parametrize(
    ("filename", "checksum"),
    [
        (
            "ge_nunley.yml",
            "66dccd2ba3b1ad91abbc6322766086a1d60039f927fcd6816213bf13df7d5997",
        ),
        (
            "aluminum_rakic.yml",
            "e93b31fe821551836e5b000949c67c633b8a1ca1f2c5549f4db0f1c03b24844d",
        ),
        (
            "tin_beliaev.yml",
            "6829a77084160e17d9351de8d7456f1bf8178bd83ad5cc61c2e3bffd44c425fb",
        ),
    ],
)
def test_reference_data_is_preserved_verbatim(filename: str, checksum: str) -> None:
    import gdsfactory.gpdk.material_cards as module

    path = Path(module.__file__).with_name("material_data") / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum


def test_lumerical_canonical_names_retain_existing_catalog_mapping() -> None:
    from gdsfactory.gpdk.simulation_settings import SIMULATION_SETTINGS_LUMERICAL_FDTD

    mapping = SIMULATION_SETTINGS_LUMERICAL_FDTD.material_name_to_lumerical
    for alias in ("si", "sio2", "sin"):
        assert mapping[MATERIAL_ALIASES[alias]] == mapping[alias]
