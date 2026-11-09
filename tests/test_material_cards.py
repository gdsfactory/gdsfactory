"""Generic PDK material coverage, provenance and legacy numerical compatibility."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from pdk_schema import Index, MaterialCard, Permittivity, Sellmeier

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
def test_legacy_helpers_remain_available(name: str, legacy: object) -> None:
    expected = {"Si": 3.475209925, "SiO2": 1.457742845, "SiN": 1.99836337}
    assert legacy(1.55) == pytest.approx(expected[name])  # type: ignore[operator]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Si", 3.475687046286856 + 0j),
        ("SiO2", 1.4440236217032607 + 0j),
        ("SiN", 1.9962797317138814 + 0j),
        ("Ge", 4.273992922541563 + 0.034421802663521725j),
        ("Aluminum", 1.5355447156902056 + 15.330400044680754j),
        ("TiN", 2.247078930025222 + 8.040529194916765j),
        ("Air", 1 + 0j),
    ],
)
def test_published_reference_indices_at_1550_nm(name: str, expected: complex) -> None:
    """Keep the public reference evaluations stable, including optical loss."""
    assert get_refractive_index(name, 1.55) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("name", ["Si", "SiO2", "SiN", "Ge", "Aluminum", "TiN"])
def test_material_tables_refuse_extrapolation(name: str) -> None:
    card = MATERIAL_CARDS[name]
    assert card.optical is not None
    model = card.optical.permittivity
    assert isinstance(model, (Index, Permittivity, Sellmeier))
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


def test_exact_card_overrides_an_inherited_alias() -> None:
    base = gf.Pdk(
        name="alias-base",
        material_cards={"Air": MATERIAL_CARDS["Air"]},
        material_aliases={"air": "Air", "ambient": "air"},
    )
    custom = MATERIAL_CARDS["Air"].model_copy(
        update={"name": "air", "info": {"distinct": True}}
    )
    child = gf.Pdk(name="alias-child", base_pdks=[base], material_cards={"air": custom})
    assert child.get_material_card("air") is custom
    assert child.get_material_card("ambient") is custom
    assert child.get_material_card("Air") is MATERIAL_CARDS["Air"]


def test_primary_germanium_extract_is_preserved() -> None:
    path = Path(gf.__file__).parent / "gpdk/material_data/ge_emminger_300k.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "cdb91c62c28d9b81552b05a0528e22e0a035ef88858650f8d94c165bf90d1fc1"
    )


@pytest.mark.parametrize(
    ("energy_ev", "real", "imaginary"),
    [
        (0.5, 16.8101094, 3.64715557e-7),
        (0.8, 18.2663541, 0.29571374),
        (1.0, 18.8456859, 0.83963444),
        (2.0, 31.2538827, 11.6236605),
    ],
)
def test_germanium_matches_author_corrected_permittivity(
    energy_ev: float,
    real: float,
    imaginary: float,
) -> None:
    """Independent anchors in Emminger's original workbook, Fit / 300 sheet."""
    from scipy.constants import c, electron_volt, h

    wavelength = h * c / electron_volt * 1e6 / energy_ev
    assert get_refractive_index("Ge", wavelength) ** 2 == pytest.approx(
        complex(real, imaginary),
        rel=3e-8,
    )


@pytest.mark.parametrize(
    ("temperature", "expected"),
    [(100, 3.4467), (293, 3.4757), (750, 3.5767)],
)
def test_silicon_matches_li_recommended_indices(
    temperature: float, expected: float
) -> None:
    """Li (1980) Table 1 at 1.55 um, rounded to the table's printed precision."""
    model = MATERIAL_CARDS["Si"].optical.permittivity
    table = model.n.data
    shape = (
        len(table.coords["wavelength"].values),
        len(table.coords["temperature"].values),
    )
    values = np.asarray(table.values).reshape(shape)
    index = values[
        table.coords["wavelength"].values.index(1.55),
        table.coords["temperature"].values.index(temperature),
    ]
    assert index == pytest.approx(expected, abs=5e-5)


@pytest.mark.parametrize("name", ["Aluminum", "TiN"])
def test_metal_table_interpolation_matches_published_equations(name: str) -> None:
    from gdsfactory.gpdk.material_models import aluminum_index, tin_index

    model = aluminum_index if name == "Aluminum" else tin_index
    wavelengths = np.linspace(0.301, 1.599, 607)
    np.testing.assert_allclose(
        get_refractive_index(name, wavelengths), model(wavelengths), rtol=2e-5
    )


def test_metal_passivity_and_primary_coefficient_conversion() -> None:
    """Re-express the paper's complex oscillator equation as independent real parts."""
    from scipy.constants import c, electron_volt, h

    wavelength = 1.31
    energy = h * c / electron_volt * 1e6 / wavelength
    parameters = {
        "Aluminum": (
            1,
            14.98**2 * 0.523,
            0.047,
            [
                (0.227 * 14.98**2, 0.162, 0.333),
                (0.05 * 14.98**2, 1.544, 0.312),
                (0.166 * 14.98**2, 1.808, 1.351),
                (0.03 * 14.98**2, 3.473, 3.382),
            ],
        ),
        "TiN": (
            3.558,
            7.44**2,
            0.416,
            [(3.819 * 4.802**2, 4.802, 2.4), (0.997 * 2.024**2, 2.024, 1.617)],
        ),
    }
    for name, (offset, plasma_squared, damping, terms) in parameters.items():
        real = offset - plasma_squared / (energy**2 + damping**2)
        imaginary = plasma_squared * damping / (energy * (energy**2 + damping**2))
        for strength, resonance, width in terms:
            difference = resonance**2 - energy**2
            denominator = difference**2 + (width * energy) ** 2
            real += strength * difference / denominator
            imaginary += strength * width * energy / denominator
        index = get_refractive_index(name, wavelength)
        assert index.real > 0 and index.imag > 0
        assert index**2 == pytest.approx(complex(real, imaginary), rel=2e-5)


def test_cards_have_primary_provenance_and_strict_validity() -> None:
    for name, card in MATERIAL_CARDS.items():
        assert card.rf is None
        assert card.optical is not None
        if name != "Air":
            assert card.optical.provenance is not None
            assert card.optical.provenance.source == "literature"
            assert card.optical.provenance.citations
            assert card.optical.permittivity.validity.on_out_of_range == "raise"


def test_lumerical_canonical_names_retain_existing_catalog_mapping() -> None:
    from gdsfactory.gpdk.simulation_settings import SIMULATION_SETTINGS_LUMERICAL_FDTD

    mapping = SIMULATION_SETTINGS_LUMERICAL_FDTD.material_name_to_lumerical
    for alias in ("si", "sio2", "sin"):
        assert mapping[MATERIAL_ALIASES[alias]] == mapping[alias]
