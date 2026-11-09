"""Generic optical cards authored from published equations and primary data."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from pdk_schema import (
    Band,
    Citation,
    Coord,
    Index,
    MaterialCard,
    Provenance,
    Regime,
    ScalarValue,
    Sellmeier,
    SellmeierTerm,
    TableData,
    TabulatedValue,
    Validity,
)

from gdsfactory.gpdk.material_models import (
    HC_EV_UM,
    LI_WAVELENGTHS_UM,
    aluminum_index,
    silicon_li_index,
    tin_index,
)

MATERIAL_ALIASES = {
    "si": "Si",
    "sio2": "SiO2",
    "sin": "SiN",
    "ge": "Ge",
    "air": "Air",
    "silicon": "Si",
    "Oxide": "SiO2",
}


def _validity(lower: float, upper: float) -> Validity:
    return Validity(
        at=None,
        over={"wavelength": Band(min=lower, max=upper, unit="um", label=None)},
        on_out_of_range="raise",
    )


def _table(values: npt.ArrayLike, coordinates: npt.ArrayLike) -> TabulatedValue:
    return TabulatedValue(
        unit="",
        data=TableData(
            dims=("wavelength",),
            coords={
                "wavelength": Coord(values=np.asarray(coordinates).tolist(), unit="um")
            },
            values=np.asarray(values).tolist(),
            attrs={},
            interp="linear",
        ),
    )


def _card(
    name: str,
    model: Index | Sellmeier,
    *,
    label: str,
    url: str | None,
    comment: str,
    doi: str | None = None,
    data_url: str | None = None,
    temperature_ref: float | None = None,
) -> MaterialCard:
    return MaterialCard(
        name=name,
        optical=Regime(
            temperature_ref=temperature_ref,
            provenance=Provenance(
                source="literature" if url else "gdsfactory",
                label=label,
                maturity="empirical" if url else "toy_model",
                citations=[
                    Citation(role="data", doi=doi, journal=None, authors=None, url=url)
                ]
                if url
                else [],
                comment=comment,
                url=url,
                data_url=data_url,
                info={},
            ),
            permittivity=model,
            conductivity=None,
            permeability=None,
            perturbations=[],
            info={},
        ),
        rf=None,
        info={"purpose": "Generic optical reference; not process-qualified."},
    )


def _sellmeier(
    terms: list[tuple[float, float]], lower: float, upper: float
) -> Sellmeier:
    return Sellmeier(
        terms=tuple(SellmeierTerm(b=strength, c_um=pole) for strength, pole in terms),
        offset=0,
        validity=_validity(lower, upper),
        variation=None,
        conductivity=None,
    )


def _silicon() -> MaterialCard:
    temperatures = np.arange(100.0, 751.0)
    values = np.column_stack(
        [
            silicon_li_index(LI_WAVELENGTHS_UM, temperature)
            for temperature in temperatures
        ]
    )
    validity = _validity(1.2, 14.0)
    validity.over["temperature"] = Band(min=100, max=750, unit="K", label=None)
    return _card(
        "Si",
        Index(
            validity=validity,
            variation=None,
            conductivity=None,
            k=None,
            n=TabulatedValue(
                unit="",
                data=TableData(
                    dims=("wavelength", "temperature"),
                    coords={
                        "wavelength": Coord(
                            values=LI_WAVELENGTHS_UM.tolist(), unit="um"
                        ),
                        "temperature": Coord(values=temperatures.tolist(), unit="K"),
                    },
                    values=values.ravel().tolist(),
                    attrs={},
                    interp="linear",
                ),
            ),
        ),
        label="Li 1980 crystalline silicon",
        doi="10.1063/1.555624",
        url="https://doi.org/10.1063/1.555624",
        data_url="https://srd.nist.gov/jpcrdreprint/1.555624.pdf",
        comment="Temperature-dependent crystalline silicon equation. Same coefficients, wavelength nodes, 1 K temperature sampling and strict ranges as gsim Si-Li-293K. Reference temperature 293 K.",
        temperature_ref=293,
    )


def _germanium() -> MaterialCard:
    path = Path(__file__).with_name("material_data") / "ge_emminger_300k.csv"
    energy, real, imaginary = np.loadtxt(path, delimiter=",", skiprows=1).T
    wavelengths = HC_EV_UM / energy[::-1]
    index = np.sqrt(real[::-1] + 1j * imaginary[::-1])
    return _card(
        "Ge",
        Index(
            n=_table(index.real, wavelengths),
            k=_table(index.imag, wavelengths),
            validity=_validity(float(wavelengths[0]), float(wavelengths[-1])),
            variation=None,
            conductivity=None,
        ),
        label="Emminger et al. 2020 bulk germanium at 300 K",
        doi="10.1116/1.5129685",
        url="https://doi.org/10.1116/1.5129685",
        data_url="http://femto.nmsu.edu/2020/Carola_2020_Ge_figures.opj",
        comment="Author workbook: Fit / 300, photon energy, eps1 and eps2. Native-oxide-corrected dielectric function, converted to passive n+i*k at each node and linearly interpolated in vacuum wavelength.",
        temperature_ref=300,
    )


def _metal(name: str) -> MaterialCard:
    if name == "Aluminum":
        wavelengths = np.geomspace(HC_EV_UM / 6, HC_EV_UM / 0.1, 4097)
        index = aluminum_index(wavelengths)
        metadata = dict(
            label="Rakic et al. 1998 aluminum Lorentz-Drude model",
            doi="10.1364/AO.37.005271",
            url="https://doi.org/10.1364/AO.37.005271",
            comment="Equation (1), Tables 1-2, Al column. Evaluated locally on a logarithmic wavelength grid over the fitted 0.1-6 eV range; n + i*k uses the passive branch.",
        )
    else:
        wavelengths = np.linspace(0.3, 1.6, 1301)
        index = tin_index(wavelengths)
        metadata = dict(
            label="Bower et al. 2020 HIPIMS TiN on silicon",
            doi="10.1021/acsami.0c10570",
            url="https://doi.org/10.1021/acsami.0c10570",
            data_url="https://shura.shu.ac.uk/27205/5/Loch_CMOS_Compatible_Desposition%28Supp%29.pdf",
            comment="Supporting Information Table S5.1, TiN / Si row. Film deposited at 300 C; optical measurement range 300-1600 nm. Standard passive Drude-Lorentz equation, sampled locally every 1 nm. Not a universal TiN process model.",
        )
    return _card(
        name,
        Index(
            n=_table(index.real, wavelengths),
            k=_table(index.imag, wavelengths),
            validity=_validity(float(wavelengths[0]), float(wavelengths[-1])),
            variation=None,
            conductivity=None,
        ),
        **metadata,
    )


MATERIAL_CARDS: dict[str, MaterialCard] = {
    "Si": _silicon(),
    "SiO2": _card(
        "SiO2",
        _sellmeier(
            [(0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161)],
            0.21,
            6.7,
        ),
        label="Malitson 1965 fused silica at 20 C",
        doi="10.1364/JOSA.55.001205",
        url="https://doi.org/10.1364/JOSA.55.001205",
        comment="Equation (1), using the same coefficients and 0.21-6.7 um reference interval as gsim SiO2-Malitson. Lossless optical-quality fused silica reference, not a process-specific deposited oxide.",
        temperature_ref=293.0,
    ),
    "SiN": _card(
        "SiN",
        _sellmeier([(3.0249, 0.1353406), (40314, 1239.842)], 0.310, 5.504),
        label="Luke et al. 2015 LPCVD silicon nitride",
        doi="10.1364/OL.40.004823",
        url="https://doi.org/10.1364/OL.40.004823",
        comment="Equation (1), wavelength converted from nm to um. LPCVD 340 nm film annealed for 3 h at 1200 C. Lossless Sellmeier fit in its published range.",
    ),
    "Ge": _germanium(),
    "Air": _card(
        "Air",
        Index(
            n=ScalarValue(value=1, unit=""),
            k=None,
            validity=None,
            variation=None,
            conductivity=None,
        ),
        label="Ideal air",
        url=None,
        comment="Lossless, nondispersive n=1 approximation.",
    ),
    "Aluminum": _metal("Aluminum"),
    "TiN": _metal("TiN"),
}


def get_refractive_index(
    name: str,
    wavelength_um: float | npt.NDArray[np.float64],
) -> complex | npt.NDArray[np.complex128]:
    """Evaluate a gpdk optical card as n + i*k without extrapolation."""
    card = MATERIAL_CARDS[MATERIAL_ALIASES.get(name, name)]
    assert card.optical is not None
    model = card.optical.permittivity
    assert isinstance(model, (Index, Sellmeier))
    wavelength = np.asarray(wavelength_um, dtype=float)
    if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0):
        raise ValueError("wavelength must be finite and positive (um)")
    if model.validity is not None and model.validity.over is not None:
        band = model.validity.over["wavelength"]
        if np.any(wavelength < band.min) or np.any(wavelength > band.max):
            raise ValueError(
                f"wavelength must be within [{band.min}, {band.max}] um for {name}"
            )

    def evaluate(value: ScalarValue | TabulatedValue | None) -> npt.NDArray[np.float64]:
        if value is None or isinstance(value, ScalarValue):
            return np.full(wavelength.shape, value.value if value else 0, dtype=float)
        values = np.asarray(value.data.values)
        if "temperature" in value.data.dims:
            temperatures = value.data.coords["temperature"].values
            values = values.reshape((-1, len(temperatures)))
            values = np.array(
                [
                    np.interp(card.optical.temperature_ref, temperatures, row)
                    for row in values
                ]
            )
        return np.asarray(
            np.interp(
                wavelength,
                value.data.coords["wavelength"].values,
                values,
            )
        )

    if isinstance(model, Sellmeier):
        squared = wavelength**2
        permittivity = (
            1
            + model.offset
            + sum(term.b * squared / (squared - term.c_um**2) for term in model.terms)
        )
        result = np.sqrt(permittivity.astype(complex))
    else:
        result = evaluate(model.n) + 1j * evaluate(model.k)
    return complex(result) if result.ndim == 0 else result
