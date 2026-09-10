# Cross sections and extrusion

`gf.CrossSection` is the union of `gf.SymmetricCrossSection`
(`kfactory.DCrossSection`) and `gf.AsymmetricCrossSection`
(`kfactory.DAsymmetricCrossSection`). Both use micrometers. Factories and
`gf.get_cross_section()` return these objects directly. Their profiles travel
with ports through GDS and OAS metadata; no separate gdsfactory profile registry
is needed to recover them.

Create a profile with `gf.cross_section.cross_section()`. Auxiliary strips use
absolute signed bounds, or `kfactory.DCrossSectionLayer`:

```python
import gdsfactory as gf

gf.gpdk.PDK.activate()
xs = gf.cross_section.cross_section(
    width=0.5, layer="WG",
    sections=[("SLAB90", -3.25, 3.25)],
    bbox_layers=["DEVREC"], bbox_offsets=[2.0],
)
sections = xs.get_sections()  # main strip first, absolute micrometer bounds
component = gf.path.straight(10).extrude(xs)
```

Each bound is snapped independently with `kcl.to_dbu()`. KLayout rounds halfway
values away from zero: at 1 nm DBU, ±0.2505 µm becomes ±251 DBU. A centered
nominal width of 0.501 µm therefore becomes 0.502 µm and remains symmetric.
Odd **realized** spans are represented by the asymmetric type. Symmetry is
tested after snapping and merging auxiliary overlaps, with the main strip kept
separate. Layer names participate in automatic names; use consistent layer
names across layouts for stable names.

Symmetric profiles store auxiliary bands relative to the core edge. A 3 µm
enclosure reaches ±3.25 µm around a 0.5 µm core and ±3.55 µm around a 1.1 µm
core. `get_sections()` resolves these bands into absolute strips. Ring bands
produce two strips; touching strips merge; fully shrunken bands disappear.
`gf.cross_section.with_width(xs, width)` explicitly preserves the auxiliary
strips' absolute bounds while replacing the main width.

Port names, port types, variable widths and offsets, simplification, insets,
and hidden strips belong to extrusion, not to a profile. Dictionaries below
select indices in `get_sections()`:

```python
component = gf.path.straight(10, npoints=101).extrude(
    xs,
    width_function=lambda t: 0.5 + t,
    ports={0: ("input", "output", "optical")},
    insets={1: (1.0, 2.0)},
    simplify=0.01,
)
```

`ports={}` suppresses all ports. Without an explicit map, the PDK's
`layer_port_types` chooses the main port type and `auxiliary_port_types` chooses
auxiliary layers with ports, keyed by `(main_layer, auxiliary_layer)` physical
layer tuples. The generic PDK enables heater contacts and slot rails this way.
Names use per-type counters (`o1`, `o2`, `e1`, `e2`, …).
Use an explicit map for device-specific contacts. An auxiliary port carries
only its own recentered strip, with its offset in the port transform. An odd
strip span uses asymmetric bounds so the port origin stays on the grid.

Transitions still use `gf.path.transition()` or `transition_asymmetric()`.
Extrusion pairs the main strips and matches auxiliary strips by layer and
signed-bound order. Use `section_pairs` for a different pairing and
`skip_transition` or `hidden` to omit selected geometry. Attach repeated
components separately with `gf.path.along_path()`.

Dynamic-width and offset extrusions attach the realized endpoint profile,
including auxiliary strips, to each main port. A taper may end at zero width
when that endpoint has no port: use a `None` endpoint name or `ports={}`.
Copying an asymmetric port
preserves its mirror state by default; connections between identical asymmetric
profiles use `mirror=True, use_mirror=True`.

## Regression changes

No GDS geometry goldens were regenerated. Settings/netlist snapshots changed
for canonical cross-section names, physical `LayerInfo` tuples, the removal of
`strip_no_ports`, and explicit extrusion parameters. Instance placements and
netlist connectivity were compared before accepting those changes.

The `rib` preset no longer stores its former 0.05 µm slab simplification.
For `ring_single(cross_section="rib")`, comparison with the original layout
found identical WG geometry and a 3.43365 µm² SLAB90 XOR. The slab area changes
from 1066.994205 to 1067.644168 µm²; after `fix_spacing(min_space=1)` the tested
area is 1070.968887 µm². `ring_single_pn` explicitly supplies the former slab
tolerance at extrusion time and matches its existing geometry golden.

## Bounding-box layers

Both kfactory cross-section types store `bbox_sections` as a mapping from
`LayerInfo` to padding in micrometers. To draw them during extrusion, use
`path.extrude(xs, add_bbox=True)` or
`gf.path.extrude(path, xs, add_bbox=True)`. The flag defaults to `False` and does
not change the profile or its port metadata. Padding surrounds all emitted
geometry, including auxiliary strips, after applying extrusion controls.
An empty extrusion produces no bbox geometry.

For a manually drawn component, use kfactory's `xs.add_bbox()` directly:

```python
xs.add_bbox(component)  # all geometry, including child instances
xs.add_bbox(component, ref=xs.layer)  # only the main layer's bounds
xs.add_bbox(component, ref=instance)  # bounds after instance placement
xs.add_bbox(component, ref=gf.kdb.DBox(0, -1, 10, 1), top=0)
```

These are alternative ways to select the reference bounds, not calls to stack.
`ref` also accepts a layer index or an integer `Box` in the target layout's dbu.
Instances (`Instance`, `DInstance`, or `VInstance`) use their transformed bounds
in their parent's coordinates; these must match the target component's coordinate
system. Cells are not accepted as `ref`; pass `cell.dbbox()` explicitly instead.
The cross section, target, and instance reference must share the same `KCLayout`
object. Equal DBUs do not make different layouts interchangeable: their layer
indices can differ. A mismatch raises `ValueError` before drawing or layer lookup.
Real cells use integer-DBU geometry; virtual cells use a separate micrometer
implementation that preserves off-grid bounds and padding overrides.
The method snapshots the reference bounds so padding does not accumulate
between layers. Optional `top`, `bottom`, `left`, and `right` override padding
for that edge on every bbox layer, in the cross section's units. Bend factories
use these overrides to retain their existing clipping. The separate
`gf.cross_section.add_bbox()` helper is removed.

For a `Component` with pending `vinsts`, bounds include approximate virtual-instance
geometry and `xs.add_bbox()` logs a warning, even with an explicit reference.
Call `component.insert_vinsts()` before `xs.add_bbox(component)` for bounds based
on materialized geometry. Adding bbox layers does not insert virtual instances
automatically. `ComponentAllAngle` uses virtual instances normally and does not
emit this warning merely because it contains them.

## Radius defaults

`validate_radius(xs, radius)` remains a function in `gf.cross_section`.
Bend and route radius overrides belong on the bend or route
call. A layout has one canonical name per profile, and different explicit
radius defaults for the same profile conflict. `metal_routing` is an alias of
the `metal3` factory. `strip_no_ports` is replaced by `strip` and `ports={}`.

Define `radius` and `radius_min` at first creation: a profile registered without
them cannot acquire them later. Width/layer-only ports and auxiliary ports use
the PDK's `port_cross_sections` mapping (physical layer tuple → factory) to
establish their process defaults immediately. Unconfigured layers produce bare
profiles with no radii. Explicit cross-section arguments are used as supplied;
the low-level `cross_section()` factory does not infer process defaults.

An intentional abrupt interface between symmetric and asymmetric doping or
cladding profiles can be placed using `gf.port.core_port()` on both connection
ports. This selects the physical core for that placement; the original ports
retain their complete profiles.
