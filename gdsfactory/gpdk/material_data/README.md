# Generic optical reference data

The generic PDK covers every material named by its layer stack with a
`pdk_schema.MaterialCard`: `Si`, `SiO2`, `SiN`, `Ge`, `Air`, `Aluminum`, and `TiN`.
These are optical examples, not process-qualified models. No RF conductivity
or permittivity is inferred from these optical measurements (`rf=None`).

The existing Si, SiO2 and SiN lossless index tables are preserved exactly,
including their 0.6–2.0 µm validity range and linear interpolation. Air is the
ideal nondispersive approximation n=1, k=0. The other materials include both
the real index n and extinction coefficient k; backends must support optical
loss to use them correctly. Material cards do not add dispersion support to
a solver that lacks it.

The following files are unmodified CC0 data from the
[refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database)
at revision `e6d15e364f70de1cad8a96a20f75689f2fba8b21`.
The upstream CC0 license is included as `LICENSE`. Interpolation is linear;
values outside a table's wavelength range raise an error.

| File | Primary reference | Wavelength (µm) | Conditions |
| --- | --- | --- | --- |
| `ge_nunley.yml` | [Nunley et al. (2016)](https://doi.org/10.1116/1.4963075) | 0.1879–2.48 | Germanium at 300 K |
| `aluminum_rakic.yml` | [Rakić (1995)](https://doi.org/10.1364/AO.34.004755) | 0.00012399–200 | Aluminum reference compilation |
| `tin_beliaev.yml` | [Beliaev et al. (2023)](https://doi.org/10.1016/j.optmat.2023.114237) | 0.21140–20 | 50 nm TiN film deposited by pulsed DC sputtering |

SHA-256 checksums:

```text
66dccd2ba3b1ad91abbc6322766086a1d60039f927fcd6816213bf13df7d5997  ge_nunley.yml
e93b31fe821551836e5b000949c67c633b8a1ca1f2c5549f4db0f1c03b24844d  aluminum_rakic.yml
6829a77084160e17d9351de8d7456f1bf8178bd83ad5cc61c2e3bffd44c425fb  tin_beliaev.yml
```
