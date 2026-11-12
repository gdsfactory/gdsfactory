# Optical references and primary data

The generic material cards are authored in `material_cards.py` and
`material_models.py` using the publications below. They are optical references,
not models qualified for a fabrication process. Air is the ideal n=1, k=0
approximation. All other cards reject wavelengths outside their stated ranges.

| Material | Primary reference and implementation | Wavelength range |
| --- | --- | --- |
| Si | [Li (1980)](https://doi.org/10.1063/1.555624), Eq. (22), crystalline silicon. Same model and sampling as gsim's Si-Li-293K; reference temperature 293 K. | 1.2-14 µm, 100-750 K |
| SiO2 | [Malitson (1965)](https://doi.org/10.1364/JOSA.55.001205), Eq. (1), optical-quality fused silica. Same coefficients and reference interval as gsim's SiO2-Malitson. | 0.21-6.7 µm |
| SiN | [Luke et al. (2015)](https://doi.org/10.1364/OL.40.004823), Eq. (1). Two-term Sellmeier; 340 nm LPCVD film annealed at 1200 °C for 3 h. | 0.310-5.504 µm |
| Ge | [Emminger et al. (2020)](https://doi.org/10.1116/1.5129685), author-provided corrected dielectric function at 300 K. | 0.5-6.23 eV, approximately 0.199-2.480 µm |
| Aluminum | [Rakić et al. (1998)](https://doi.org/10.1364/AO.37.005271), Eq. (1), Tables 1-2, Al column. Locally evaluated Lorentz-Drude model. | 0.1-6 eV, approximately 0.207-12.398 µm |
| TiN | [Bower et al. (2020)](https://doi.org/10.1021/acsami.0c10570), [Supporting Information](https://shura.shu.ac.uk/27205/5/Loch_CMOS_Compatible_Desposition%28Supp%29.pdf), Table S5.1, TiN on Si. Film deposited by HIPIMS at 300 °C. | 0.300-1.600 µm |

Si uses a native two-dimensional pdk-schema Index table over wavelength and
temperature, evaluated from Li's equation at the same 35 wavelength nodes and
1 K temperature intervals as gsim. SiO2 and SiN use native Sellmeier declarations,
matching gsim's Malitson and Luke cards. Their optical loss is approximated as
zero within the stated transparent ranges. Metal n,k
tables are generated locally from the implemented equations, with linear
interpolation on 4097 logarithmic wavelength samples for Aluminum and a 1 nm
grid for TiN. Tests check interpolation against the equations. Optical loss is
positive k in n+i*k, using the exp(-i*omega*t) convention. All photon energies
and linewidths in the metal equations are in eV; no frequency-unit conversion
is implicit. The TiN implementation uses the standard passive Drude-Lorentz
denominators with the published fitted coefficients.

## Germanium extract

`ge_emminger_300k.csv` contains 574 numerical rows extracted directly from the
[authors' Origin workbook](http://femto.nmsu.edu/2020/Carola_2020_Ge_figures.opj).
The source is workbook **Fit**, worksheet **300**, columns **A** (photon energy,
eV), **B** (epsilon1) and **C** (epsilon2), whose column comments identify 300 K.
The separate **Data** workbook contains pseudodielectric data and is not used.
The published analysis corrects for the native oxide layer.

Extraction used the open source [liborigin](https://github.com/gerlachs/liborigin)
reader. Only positive-energy rows were retained, sorted by increasing energy,
and written with ten significant digits. Empty Origin cells were removed.
No digitization of plots or fitting to database copies was performed. At card
construction, energy is converted to vacuum wavelength with hc=1.2398419843320026
eV µm. The passive n+i*k is obtained by taking the square root of epsilon1+i*epsilon2
at each measured node; n and k are linearly interpolated in wavelength.

SHA-256:

```text
9f4f1097d3327880f80db906ee4cdc84039e113bdb74df5f96e989029f962822  Carola_2020_Ge_figures.opj
cdb91c62c28d9b81552b05a0528e22e0a035ef88858650f8d94c165bf90d1fc1  ge_emminger_300k.csv
```

Only the numerical extract is bundled. The source documents, workbook, and
reader implementation are not distributed with GDSFactory. These cards do
not infer RF properties from optical measurements (`rf=None`).
