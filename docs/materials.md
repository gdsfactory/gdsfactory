# Generic PDK materials

The generic PDK registers typed [pdk-schema](https://github.com/gdsfactory/pdk-schema)
optical material cards for every material in its layer stack. Canonical names
are `Si`, `SiO2`, `SiN`, `Ge`, `Air`, `Aluminum`, and `TiN`.

```python
import gdsfactory as gf
from gdsfactory.gpdk.material_cards import get_refractive_index

pdk = gf.gpdk.get_generic_pdk()
card = pdk.get_material_card("Si")
serialized = card.model_dump_json()
index = get_refractive_index("Si", 1.55)  # wavelength in µm; returns n + i*k
```

`Pdk.material_cards` maps canonical names to `MaterialCard` instances.
`Pdk.material_aliases` maps compatibility names to those canonical names.
Both registries are inherited from base PDKs; derived PDK entries take precedence.
Use `get_material_card(name)` to resolve an alias. Exact cards take precedence
over inherited aliases. Aliases remain case sensitive;
the generic PDK retains `si`, `sio2`, `sin`, `ge`, `air`, `silicon`, and `Oxide`.

The cards are authored directly from published equations and author-provided
data. Si uses Li's temperature-dependent model with a 293 K reference, SiO2 uses Malitson's fused
silica equation, and SiN uses Luke et al.'s LPCVD film equation. Ge uses
Emminger et al.'s corrected complex dielectric function at 300 K. Aluminum and
TiN tables are generated locally from published Drude-Lorentz coefficients.
Air is the ideal n=1, k=0 approximation. See the
[sources, conditions, wavelength limits and data extraction](https://github.com/gdsfactory/gdsfactory/tree/main/gdsfactory/gpdk/material_data).

These references are not qualified for a fabrication process. In particular,
deposited oxides and nitrides can differ from the cited samples. Lossless
Sellmeier fits do not model absorption; Ge and metals include optical loss.
Backends must support the relevant model and loss. Cards reject extrapolation.
Si covers 1.2-14 µm and 100-750 K; the selected TiN reference covers 0.3-1.6 µm.
Si, SiO2 and SiN adopt gsim's Li, Malitson and Luke models. Li is selected over
the Salzberg silicon reference so the card also supports 1.31 µm simulations.

`materials_index` and its aliases now evaluate these cards. This replaces the
older generic Si/SiO2/SiN interpolation tables in that registry, so numerical
results and supported wavelength ranges can change. The standalone `si`,
`sio2`, and `sin` functions remain available with their previous values for
callers that require the legacy approximations.

Simulation tools can discover the PDK cards without user-supplied index overrides.
The cards specify material responses; the layer stack and layout specify the
physical regions. An unused oxide layer declaration does not create oxide
geometry. These cards do not specify RF models.
