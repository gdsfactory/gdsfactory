"""Transformer components PDK."""

import gdsfactory as gf
from gdsfactory import Component
from gdsfactory import component_functions as cf
from gdsfactory.component_functions.analog.transformers import (
    get_extended_layer_stack,
)
from gdsfactory.typings import ComponentSpec, LayerSpec, LayerSpecs

__all__ = [
    "get_extended_layer_stack",
    "stacked_transformer",
    "symmetric_transformer",
    "transformer_concentric_secondary",
    "via3",
]


# Symmetric transformer


@gf.cell_with_module_name(tags=["analog"])
def symmetric_transformer(
    d_out: float = 150.0,
    N1: int = 2,
    N2: int = 3,
    sides: int = 8,
    width: float = 7.0,
    spacing: float = 2.0,
    center_tap_primary: bool = False,
    center_tap_secondary: bool = False,
    via_extent: float | None = None,
    port_spacing: float | None = None,
    via: ComponentSpec = "via2",
    resistance: float = 0.5777,
    inductance: float = 33.303e-12,
    add_pgs: bool = False,
    pgs_diameter: float = 180.0,
    pgs_width: float = 4.0,
    pgs_spacing: float = 2.0,
    layer_winding: LayerSpec = "M3",
    layer_underpass: LayerSpec = "M2",
    layers_pgs: LayerSpecs = ("M1",),
) -> Component:
    """Symmetric (interleaved) transformer.

    Two interleaved windings (primary: N1 turns, secondary: N2 turns) share
    the same octagonal spiral, alternating turn-by-turn. Crossings are
    routed on layer_underpass wherever a turn boundary needs to jump
    between quadrants; bridges wire together the top/bottom/left/right
    quadrant segments directly on layer_winding.

    Args:
        d_out: Outer diameter of the winding structure, in micrometers.
        N1: Number of turns in the primary winding.
        N2: Number of turns in the secondary winding.
        sides: Number of polygon sides per full turn (8 = octagonal).
            Must be a multiple of 4 (quadrant angle lists use sides // 4).
        width: Metal trace width in micrometers.
        spacing: Gap between adjacent turns in micrometers.
        center_tap_primary: When True, add a center-tap bridge (and CT1 or
            CT2 port, depending on parity) for the primary winding.
        center_tap_secondary: When True, add a center-tap bridge (and CT1
            or CT2 port, depending on parity) for the secondary winding.
        via_extent: Length crossing/bridge routes extend past their
            crossing box on layer_underpass, and the box size used to size
            the crossing/centertap via arrays. If None, it's derived from
            the chosen via's own geometry the same way spiral_inductor and
            symmetric_inductor derive their "extend" value.
        port_spacing: Horizontal spacing of the differential port pairs.
        via: via ComponentSpec connecting winding <-> crossing/centertap.
        resistance: Series resistance in ohms, stored as metadata only.
        inductance: Inductance in henries, stored as metadata only.
        add_pgs: When True, add a patterned ground shield on layers_pgs.
        pgs_diameter: Bounding size D of the ground shield square, in micrometers.
        pgs_width: Strip width w of each ground shield finger, in micrometers.
        pgs_spacing: Gap s between adjacent ground shield fingers, in micrometers.
        layer_winding: Metal layer for the main winding (top metal).
        layer_underpass: Metal layer for crossings and center-tap bridges (one layer below layer_winding).
        layers_pgs: Layers on which the patterned ground shield is drawn,
            kept separate from both layer_winding and layer_underpass

    Returns:
        Component with 4-6 ports:
          P1+ / P1-  ->  primary differential terminals   (bottom, layer_winding)
          P2+ / P2-  ->  secondary differential terminals (top, layer_winding)
          CT1        ->  present if a center tap lands on the bottom side
          CT2        ->  present if a center tap lands on the top side
    """
    return cf.symmetric_transformer(
        d_out=d_out,
        N1=N1,
        N2=N2,
        sides=sides,
        width=width,
        spacing=spacing,
        center_tap_primary=center_tap_primary,
        center_tap_secondary=center_tap_secondary,
        via_extent=via_extent,
        port_spacing=port_spacing,
        via=via,
        resistance=resistance,
        inductance=inductance,
        add_pgs=add_pgs,
        pgs_diameter=pgs_diameter,
        pgs_width=pgs_width,
        pgs_spacing=pgs_spacing,
        layer_winding=layer_winding,
        layer_underpass=layer_underpass,
        layers_pgs=layers_pgs,
    )


# Stacked transformer
# ---------------------------------------------------------------------------
# 4th metal (M4) + VIA3, used by stacked_transformer's DEFAULT full-isolation config.
# The gdsfactory generic PDK only has 3 real metals (M1/M2/M3).
# M4_LAYER/VIA3_LAYER are synthetic: valid GDS layers (KLayout accepts any registered layer number)
# but NOT part of this PDK's real, fabricatable metal stack or its get_layer_stack() z-geometry.
#
# *** SIMULATION PIPELINE WARNING ***
# If you're feeding a stacked_transformer() default-config component into
# the gsim/Palace EM simulation pipeline, sim.set_stack(substrate_thickness=...)
# will NOT know M4 has any zmin/thickness/material, since get_layer_stack() has no metal4 entry.
# You MUST instead call:
#     sim.set_stack(stack=get_extended_layer_stack(), substrate_thickness=...)
# using get_extended_layer_stack() so the mesher has real z-geometry for M4/VIA3.
# This only matters for simulation, plain GDS layout/export works fine with the defaults as-is.
# ---------------------------------------------------------------------------


@gf.cell
def via3(
    size: tuple[float, float] = (0.7, 0.7),
    enclosure: float = 1.0,
    pitch: float = 2.0,
) -> Component:
    """Via connecting M3 <-> M4, mirroring via1/via2/viac's shape (0.7x0.7um squares, 1um enclosure, 2um pitch) but on VIA3_LAYER.

    Only meaningful once M4 also has real z-geometry — see
    get_extended_layer_stack() for the matching LayerStack entry.

    Args:
        size: (width, height) of the via square, in um.
        enclosure: metal enclosure around the via, in um.
        pitch: via array pitch, in um.
    """
    return cf.via3(
        size=size,
        enclosure=enclosure,
        pitch=pitch,
    )


@gf.cell_with_module_name(tags=["analog"])
def stacked_transformer(
    d_out: float = 150.0,
    N1: int = 3,
    N2: int = 3,
    sides: int = 8,
    width: float = 10.0,
    spacing: float = 2.0,
    center_tap_primary: bool = False,
    center_tap_secondary: bool = False,
    via_extent: float | None = None,
    port_spacing: float | None = None,
    via_primary: ComponentSpec = "via3",
    via_secondary: ComponentSpec = "via1",
    resistance: float = 0.5777,
    inductance: float = 33.303e-12,
    add_pgs: bool = False,
    pgs_diameter: float = 180.0,
    pgs_width: float = 4.0,
    pgs_spacing: float = 2.0,
    # Defaults use the synthetic M4/VIA3 layers for full 4-metal isolation —
    # see the module-level comment above M4_LAYER for the SIMULATION
    # PIPELINE WARNING: use get_extended_layer_stack() with sim.set_stack(),
    # not the PDK default, when meshing a component built with these.
    layer_winding_primary: LayerSpec | None = None,
    layer_crossing_primary: LayerSpec = "M3",
    layer_winding_secondary: LayerSpec = "M2",
    layer_crossing_secondary: LayerSpec = "M1",
    layers_pgs: LayerSpecs = (),
) -> Component:
    """Stacked transformer.

    The primary winding sits on layer_winding_primary with
    its crossings on layer_crossing_primary (connected by via_primary);
    the secondary winding sits on layer_winding_secondary with its
    crossings on layer_crossing_secondary (connected by via_secondary),
    mirrored to sit on the opposite side of d_out so the two windings
    stack vertically over the same footprint.

    LAYER STACK: this gdsfactory generic PDK only has 3 real metals
    (M1/M2/M3). A fully isolated stacked transformer needs 4 independent
    metals (primary winding, primary crossing, secondary winding,
    secondary crossing), so the defaults here use an EXTRA, non-native
    4th metal registered via gf.kcl.layer() at import time (M4_LAYER,
    layer (53, 0)) with its own via (via3(), VIA3_LAYER, layer (48, 0))
    connecting it to M3. This gives full primary/secondary isolation by
    default:
        layer_winding_primary=M4_LAYER, layer_crossing_primary="M3", via_primary=via3
        layer_winding_secondary="M2",   layer_crossing_secondary="M1", via_secondary="via1"

    M4_LAYER/VIA3_LAYER are valid GDS layers but are NOT part of this PDK's real,
    fabricatable metal stack. If you need this component to be simulation-ready
    (Palace/gsim meshing) ortape-out-accurate, call get_extended_layer_stack()
    instead of the PDK's default stack, e.g. sim.set_stack(stack=get_extended_layer_stack()).

    If your ACTUAL target PDK has a genuine 4th metal, pass its real
    layer name/via in place of M4_LAYER/via3 instead of relying on this
    synthetic one.

    layers_pgs defaults to an empty tuple: even with M4 in play, all of
    M1/M2/M3/M4 are live signal layers in the default configuration, so
    there's still no obviously-safe spare metal for a patterned ground
    shield — pass an explicit LayerSpecs of your own (accepting whatever
    coupling that implies) if you need one anyway.

    Args:
        d_out: Outer diameter of each winding, in micrometers.
        N1: Number of turns in the primary winding.
        N2: Number of turns in the secondary winding.
        sides: Number of polygon sides per full turn (8 = octagonal).
        width: Metal trace width in micrometers.
        spacing: Gap between adjacent turns in micrometers.
        center_tap_primary: When True, add a center-tap bridge and CT_P
            port to the primary winding.
        center_tap_secondary: When True, add a center-tap bridge and CT_S
            port to the secondary winding.
        via_extent: Length crossing routes extend past their crossing box,
            and the box size used to size the crossing/centertap via
            arrays, for both halves. If None, it's derived independently
            for each half from that half's own via geometry (same
            derivation spiral_inductor/symmetric_inductor use).
        port_spacing: Horizontal spacing of both differential port pairs.
            Defaults to spacing.
        via_primary: via ComponentSpec connecting layer_winding_primary <-> layer_crossing_primary.
        via_secondary: via ComponentSpec connecting layer_winding_secondary <-> layer_crossing_secondary.
        resistance: Series resistance in ohms, stored as metadata only.
        inductance: Inductance in henries, stored as metadata only.
        add_pgs: When True, add a patterned ground shield on layers_pgs.
            See the layer-stack caveat above before enabling this.
        pgs_diameter: Bounding size D of the ground shield square, in micrometers.
        pgs_width: Strip width w of each ground shield finger, in micrometers.
        pgs_spacing: Gap s between adjacent ground shield fingers, in micrometers.
        layer_winding_primary: Metal layer for the primary winding.
        layer_crossing_primary: Metal layer for the primary crossings.
        layer_winding_secondary: Metal layer for the secondary winding.
        layer_crossing_secondary: Metal layer for the secondary crossings.
        layers_pgs: Layers on which the patterned ground shield is drawn.

    Returns:
        Component with 4-6 ports:
          P+ / P-    ->  primary differential terminals   (layer_winding_primary)
          S+ / S-    ->  secondary differential terminals (layer_winding_secondary)
          CT_P       ->  present if center_tap_primary=True
          CT_S       ->  present if center_tap_secondary=True
    """
    return cf.stacked_transformer(
        d_out=d_out,
        N1=N1,
        N2=N2,
        sides=sides,
        width=width,
        spacing=spacing,
        center_tap_primary=center_tap_primary,
        center_tap_secondary=center_tap_secondary,
        via_extent=via_extent,
        port_spacing=port_spacing,
        via_primary=via_primary,
        via_secondary=via_secondary,
        resistance=resistance,
        inductance=inductance,
        add_pgs=add_pgs,
        pgs_diameter=pgs_diameter,
        pgs_width=pgs_width,
        pgs_spacing=pgs_spacing,
        layer_winding_primary=layer_winding_primary,
        layer_crossing_primary=layer_crossing_primary,
        layer_winding_secondary=layer_winding_secondary,
        layer_crossing_secondary=layer_crossing_secondary,
        layers_pgs=layers_pgs,
    )


# Concentric (single-turn, coplanar) 1:1 transformer


@gf.cell
def transformer_concentric_secondary(
    width: float = 3.0,
    space: float = 3.1,
    diameter: float = 50.0,
    layer_metal: LayerSpec = "M2",
    layer_jumper: LayerSpec = "M1",
    via: ComponentSpec = "via1",
    via_size: float = 3.0,
) -> Component:
    """Single-turn octagonal coil, the secondary of transformer_concentric.

    The two leads jump to a lower metal (layer_jumper) right where
    they meet the coil body, so the leads can be routed straight out
    past an outer coil without shorting it.

    Args:
        width: Metal trace width in micrometers.
        space: Space between the coil and the leads/gap, in micrometers.
        diameter: Coil diameter in micrometers.
        layer_metal: Layer for the coil body.
        layer_jumper: Layer for the two leads (must be via-connectable
            to layer_metal).
        via: via ComponentSpec connecting layer_jumper <-> layer_metal.
        via_size: Side length of the square via_stack junction pad, in
            micrometers. via_stack() requires this to be large enough to
            enclose at least one via square with its enclosure margin —
            too small raises a ValueError from via_stack() itself.

    Returns:
        Component with ports P1, P2 on layer_jumper.
    """
    return cf.transformer_concentric_secondary(
        width=width,
        space=space,
        diameter=diameter,
        layer_metal=layer_metal,
        layer_jumper=layer_jumper,
        via=via,
        via_size=via_size,
    )


@gf.cell_with_module_name(tags=["analog"])
def transformer_concentric(
    width_primary: float = 3.0,
    width_secondary: float = 3.0,
    space: float = 3.1,
    coupling_gap: float = 4.0,
    diameter_outer: float = 80.0,
    layer_primary: LayerSpec = "M3",
    layer_secondary: LayerSpec = "M2",
    layer_secondary_jumper: LayerSpec = "M1",
    via_secondary: ComponentSpec = "via1",
    via_size: float | None = None,
    layer_inductor: LayerSpec = "M1",
    layers_no_fill: LayerSpecs = ("DEVREC", "NO_TILE_SI"),
    add_pgs: bool = False,
    pgs_diameter: float = 120.0,
    pgs_width: float = 4.0,
    pgs_spacing: float = 2.0,
    layers_pgs: LayerSpecs = (),
) -> Component:
    """Concentric, coplanar 1:1 transformer (single-turn coils).

    Primary: standard inductor(), outer ring, on layer_primary.
    Secondary: transformer_concentric_secondary(), inner ring, whose
    leads are drawn on layer_secondary_jumper instead of layer_secondary, so they pass
    underneath the primary ring without colliding — a via_stack connects
    each lead to the coil body right where they meet.


    Args:
        width_primary: Primary coil trace width, in micrometers.
        width_secondary: Secondary coil trace width, in micrometers.
        space: Space between adjacent turns/leads, in micrometers
            (shared by both coils, matching the original).
        coupling_gap: Radial gap between the primary's inner edge and
            the secondary's outer edge, in micrometers.
        diameter_outer: Primary coil's outer diameter, in micrometers.
        layer_primary: Metal layer for the primary coil.
        layer_secondary: Metal layer for the secondary coil body.
        layer_secondary_jumper: Metal layer for the secondary's leads (via-connected to layer_secondary).
        via_secondary: via ComponentSpec connecting layer_secondary_jumper <-> layer_secondary.
        via_size: Side length of the secondary's via_stack junction pads,
            in micrometers. Defaults to width_secondary if None — bump
            this up if via_stack() raises an enclosure ValueError.
        layer_inductor: Marker layer for the outer IND-style polygon
            drawn around each coil (matches inductor()'s own
            layer_inductor role).
        layers_no_fill: Layers excluded from metal fill, drawn under
            the same outer marker polygon.
        add_pgs: When True, add a patterned ground shield on layers_pgs.
        pgs_diameter: Bounding size D of the ground shield square, in micrometers.
        pgs_width: Strip width w of each ground shield finger, in micrometers.
        pgs_spacing: Gap s between adjacent ground shield fingers, in micrometers.
        layers_pgs: Layers on which the patterned ground shield is drawn.
            Empty by default: layer_primary/layer_secondary/
            layer_secondary_jumper already consume all 3 of this PDK's
            real metals in the default configuration, so there's no
            obviously-safe spare layer — pass one explicitly (accepting
            whatever coupling that implies) if you need a shield anyway.

    Returns:
        Component with ports P1, P2 (primary, on layer_primary) and
        S1, S2 (secondary, on layer_secondary_jumper).
    """
    return cf.analog.transformers.transformer_concentric(
        width_primary=width_primary,
        width_secondary=width_secondary,
        space=space,
        coupling_gap=coupling_gap,
        diameter_outer=diameter_outer,
        layer_primary=layer_primary,
        layer_secondary=layer_secondary,
        layer_secondary_jumper=layer_secondary_jumper,
        via_secondary=via_secondary,
        via_size=via_size,
        layer_inductor=layer_inductor,
        layers_no_fill=layers_no_fill,
        add_pgs=add_pgs,
        pgs_diameter=pgs_diameter,
        pgs_width=pgs_width,
        pgs_spacing=pgs_spacing,
        layers_pgs=layers_pgs,
    )


if __name__ == "__main__":
    c = symmetric_transformer()
    c.show()
