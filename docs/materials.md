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
Existing Si, SiO2 and SiN `materials_index` callables and their numerical values
are unchanged. Canonical names are also available in `materials_index`.

The cards are generic optical examples, not fabrication-qualified models. Si,
SiO2 and SiN retain the existing lossless tables over 0.6–2.0 µm; Air is n=1, k=0.
Ge, Aluminum and TiN include tabulated loss with citations and measured wavelength
limits. The TiN reference describes one sputtered film, not every TiN process.
Tables refuse extrapolation. See the bundled
[reference data and conditions](https://github.com/gdsfactory/gdsfactory/tree/main/gdsfactory/gpdk/material_data).

Simulation tools can discover the PDK cards without user-supplied index overrides.
When project cards take precedence over a tool's built-in fallback, the PDK's
existing optical indices become authoritative and can differ from that fallback.
A solver must support the card's optical model and loss. These cards do not
specify RF models or change which physical regions are present in a component;
for example, an unused oxide layer declaration does not create oxide geometry.
