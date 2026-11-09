"""Optical reference cards for every material in the generic layer stack.

These are generic examples, not fabrication-qualified material models. Existing
Si, SiO2 and SiN index tables are retained; other tables include optical loss.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml
from pdk_schema import (
    Band,
    Citation,
    Coord,
    Index,
    MaterialCard,
    Provenance,
    Regime,
    ScalarValue,
    TableData,
    TabulatedValue,
    Validity,
)

from gdsfactory.gpdk.material_indices import (
    refractive_indices_nitride,
    refractive_indices_oxide,
    refractive_indices_silicon,
    wavelengths,
)

_DATA = Path(__file__).with_name("material_data")
_REVISION = "e6d15e364f70de1cad8a96a20f75689f2fba8b21"

MATERIAL_ALIASES = {
    "si": "Si",
    "sio2": "SiO2",
    "sin": "SiN",
    "ge": "Ge",
    "air": "Air",
    "silicon": "Si",
    "Oxide": "SiO2",
}


def _table(values: list[float], coordinates: list[float]) -> TabulatedValue:
    return TabulatedValue(
        unit="",
        data=TableData(
            dims=("wavelength",),
            coords={"wavelength": Coord(values=coordinates, unit="um")},
            values=values,
            attrs={},
            interp="linear",
        ),
    )


def _card(
    name: str,
    n: ScalarValue | TabulatedValue,
    k: ScalarValue | TabulatedValue,
    provenance: Provenance,
    coordinates: list[float] | None = None,
    temperature_ref: float | None = None,
) -> MaterialCard:
    validity = None
    if coordinates is not None:
        validity = Validity(
            at=None,
            over={
                "wavelength": Band(
                    min=coordinates[0], max=coordinates[-1], unit="um", label=None
                )
            },
            on_out_of_range="raise",
        )
    return MaterialCard(
        name=name,
        optical=Regime(
            temperature_ref=temperature_ref,
            provenance=provenance,
            permittivity=Index(
                validity=validity, variation=None, conductivity=None, n=n, k=k
            ),
            conductivity=None,
            permeability=None,
            perturbations=[],
            info={},
        ),
        rf=None,
        info={"purpose": "Generic optical simulation example; not process-qualified."},
    )


def _example_provenance(label: str, comment: str) -> Provenance:
    return Provenance(
        source="gdsfactory",
        label=label,
        maturity="toy_model",
        citations=[],
        comment=comment,
        url=None,
        data_url=None,
        info={},
    )


def _reference_card(
    name: str,
    filename: str,
    source_path: str,
    doi: str,
    label: str,
    temperature_ref: float | None = None,
) -> MaterialCard:
    document = yaml.safe_load((_DATA / filename).read_text())
    rows = [
        list(map(float, row.split()))
        for row in document["DATA"][0]["data"].splitlines()
        if row.strip()
    ]
    coordinates, n, k = map(list, zip(*rows, strict=True))
    return _card(
        name,
        _table(n, coordinates),
        _table(k, coordinates),
        Provenance(
            source="literature",
            label=label,
            maturity="empirical",
            citations=[
                Citation(
                    role="data",
                    doi=doi,
                    journal=None,
                    authors=None,
                    url=f"https://doi.org/{doi}",
                )
            ],
            comment=document.get("COMMENTS"),
            url=f"https://doi.org/{doi}",
            data_url=f"https://github.com/polyanskiy/refractiveindex.info-database/blob/{_REVISION}/{source_path}",
            info={"license": "CC0-1.0", "database_revision": _REVISION},
        ),
        coordinates,
        temperature_ref,
    )


MATERIAL_CARDS: dict[str, MaterialCard] = {
    name: _card(
        name,
        _table(values, wavelengths),
        ScalarValue(value=0, unit=""),
        _example_provenance(
            f"Generic {name} index table",
            "Existing generic PDK lossless index table; linear interpolation.",
        ),
        wavelengths,
    )
    for name, values in {
        "Si": refractive_indices_silicon,
        "SiO2": refractive_indices_oxide,
        "SiN": refractive_indices_nitride,
    }.items()
}
MATERIAL_CARDS.update(
    {
        "Air": _card(
            "Air",
            ScalarValue(value=1, unit=""),
            ScalarValue(value=0, unit=""),
            _example_provenance(
                "Ideal air", "Lossless, nondispersive n=1 approximation to air."
            ),
        ),
        "Ge": _reference_card(
            "Ge",
            "ge_nunley.yml",
            "database/data/main/Ge/nk/Nunley.yml",
            "10.1116/1.4963075",
            "Nunley et al. 2016 germanium",
            300,
        ),
        "Aluminum": _reference_card(
            "Aluminum",
            "aluminum_rakic.yml",
            "database/data/main/Al/nk/Rakic.yml",
            "10.1364/AO.34.004755",
            "Rakic 1995 aluminum",
        ),
        "TiN": _reference_card(
            "TiN",
            "tin_beliaev.yml",
            "database/data/main/TiN/nk/Beliaev-sputtering.yml",
            "10.1016/j.optmat.2023.114237",
            "Beliaev et al. 2023 sputtered 50 nm TiN film",
        ),
    }
)


def get_refractive_index(
    name: str,
    wavelength_um: float | npt.NDArray[np.float64],
) -> complex | npt.NDArray[np.complex128]:
    """Evaluate a generic optical card as n + i*k, refusing extrapolation."""
    card = MATERIAL_CARDS[MATERIAL_ALIASES.get(name, name)]
    assert card.optical is not None
    model = card.optical.permittivity
    assert isinstance(model, Index)
    wavelength = np.asarray(wavelength_um, dtype=float)
    if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0):
        raise ValueError("wavelength must be finite and positive (um)")
    if model.validity is not None and model.validity.over is not None:
        band = model.validity.over["wavelength"]
        if np.any(wavelength < band.min) or np.any(wavelength > band.max):
            raise ValueError(
                f"wavelength must be within [{band.min}, {band.max}] um for {name}"
            )

    def evaluate(value: ScalarValue | TabulatedValue) -> npt.NDArray[np.float64]:
        if isinstance(value, ScalarValue):
            return np.full(wavelength.shape, value.value)
        return np.asarray(
            np.interp(
                wavelength, value.data.coords["wavelength"].values, value.data.values
            )
        )

    assert isinstance(model.n, (ScalarValue, TabulatedValue))
    assert isinstance(model.k, (ScalarValue, TabulatedValue))
    result = evaluate(model.n) + 1j * evaluate(model.k)
    return complex(result) if result.ndim == 0 else result
