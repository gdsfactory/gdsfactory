"""Inductor components PDK."""

import gdsfactory as gf
from gdsfactory import Component
from gdsfactory import component_functions as cf
from gdsfactory.typings import ComponentSpec, LayerSpec, LayerSpecs

from .._schematic import inductor_schematic

__all__ = ["inductor", "spiral_inductor", "symmetric_inductor"]


def inductor_min_diameter(width: float, space: float, turns: int, grid: float) -> float:
    """Calculate minimum diameter for inductor.

    Args:
        width: Width of the inductor trace in micrometers.
        space: Space between turns in micrometers.
        turns: Number of turns.
        grid: Grid resolution.

    Returns:
        Minimum diameter in micrometers.
    """
    min_d = 2 * turns * (width + space) + 4 * width
    return round(min_d / grid) * grid


@gf.cell_with_module_name(schematic_function=inductor_schematic, tags=["analog"])
def inductor(
    width: float = 2.0,
    space: float = 2.1,
    diameter: float = 25.35,
    resistance: float = 0.5777,
    inductance: float = 33.303e-12,
    turns: int = 1,
    layer_metal: LayerSpec = "M3",
    layer_inductor: LayerSpec = "M1",
    layer_metal_pin: LayerSpec = "WG_PIN",
    layers_no_fill: LayerSpecs = ("DEVREC", "NO_TILE_SI"),
) -> Component:
    """Create a 2-turn inductor.

    Args:
        width: Width of the inductor trace in micrometers.
        space: Space between turns in micrometers.
        diameter: Inner diameter in micrometers.
        resistance: Resistance in ohms.
        inductance: Inductance in henries.
        turns: Number of turns (default 1 for inductor2).
        layer_metal: Layer for the metal trace.
        layer_inductor: Layer for the inductor region.
        layer_metal_pin: Layer for the metal pins.
        layers_no_fill: Layers to exclude from fill.

    Returns:
        Component with inductor layout.
    """
    return cf.inductor(
        width=width,
        space=space,
        diameter=diameter,
        resistance=resistance,
        inductance=inductance,
        turns=turns,
        layer_metal=layer_metal,
        layer_inductor=layer_inductor,
        layer_metal_pin=layer_metal_pin,
        layers_no_fill=layers_no_fill,
    )


@gf.cell_with_module_name(tags=["analog"])
def spiral_inductor(
    d_out: float = 130.0,
    N: int = 3,
    sides: int = 8,
    width: float = 10.0,
    spacing: float = 4.0,
    aspect_ratio: float = 1.0,
    port_side: str = "same",
    add_pgs: bool = False,
    pgs_diameter: float = 150.0,
    pgs_width: float = 2.0,
    pgs_spacing: float = 1.0,
    via: ComponentSpec = "via2",
    resistance: float = 0.5777,
    inductance: float = 33.303e-12,
    layer_winding: LayerSpec = "M3",
    layer_underpass: LayerSpec = "M2",
    layers_pgs: LayerSpecs = ("M1",),
) -> Component:
    """Polygonal spiral inductor.

    Args:
        d_out: Outer diameter of the spiral in micrometers.
        N: Number of complete turns.
        sides: Number of polygon sides per full turn (8 = octagonal).
        width: Metal trace width in micrometers.
        spacing: Gap between adjacent turns in micrometers.
        aspect_ratio: Y-axis scale factor for non-square spirals (1.0 = symmetric).
        port_side: ``"same"`` keeps both ports on the same side;
            ``"opposite"`` places them on opposite sides.
        add_pgs: When True, add a patterned ground shield on layers_pgs.
        pgs_diameter: Bounding size D of the ground shield square, in micrometers.
        pgs_width: Strip width w of each ground shield finger, in micrometers.
        pgs_spacing: Gap s between adjacent ground shield fingers, in micrometers.
        via: via ComponentSpec connecting winding <-> underpass.
        resistance: Series resistance in ohms, stored as metadata only.
        inductance: Inductance in henries, stored as metadata only.
        layer_winding: Metal layer for the main spiral winding.
        layer_underpass: Metal layer for the inner-terminal underpass bridge (one layer below layer_winding).
        layers_pgs: Layers on which the patterned ground shield is drawn,
            kept separate from both layer_winding and layer_underpass
            since the underpass carries a live signal and shouldn't s
            hare a layer with a grounded shield.

    Returns:
        Component with 2 RF ports:
          P1  ->  entry terminal  (layer_winding)
          P2  ->  exit terminal   (layer_underpass)
    """
    return cf.analog.inductors.spiral_inductor(
        d_out=d_out,
        N=N,
        sides=sides,
        width=width,
        spacing=spacing,
        aspect_ratio=aspect_ratio,
        port_side=port_side,
        add_pgs=add_pgs,
        pgs_diameter=pgs_diameter,
        pgs_width=pgs_width,
        pgs_spacing=pgs_spacing,
        via=via,
        resistance=resistance,
        inductance=inductance,
        layer_winding=layer_winding,
        layer_underpass=layer_underpass,
        layers_pgs=layers_pgs,
    )


# Symmetric (differential) inductor


@gf.cell_with_module_name(tags=["analog"])
def symmetric_inductor(
    d_out: float = 150.0,
    N: int = 3,
    sides: int = 8,
    width: float = 10.0,
    spacing: float = 2.0,
    center_tap: bool = False,
    via_extent: float | None = None,
    port_spacing: float | None = None,
    aspect_ratio: float = 1.0,
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
    """Symmetric (differential) spiral inductor.

    Args:
        d_out: Outer diameter of the two-lobe structure, in micrometers.
        N: Number of complete windings per side.
        sides: Number of polygon sides per full turn (8 = octagonal).
        width: Metal trace width in micrometers.
        spacing: Gap between adjacent turns in micrometers.
        center_tap: When True, add a center-tap bridge (and CT port)
            routed through layer_underpass with its own via connection
            to the winding.
        via_extent: Length the crossing route extends past the crossing
            box on layer_underpass, and the box size used to size the
            crossing/centertap via arrays. If None, it's derived from the
            chosen via's own geometry.
        port_spacing: Horizontal spacing of the differential ports P1/P2.
        aspect_ratio: Y-axis scale factor for non-square windings
        via: via ComponentSpec connecting winding <-> crossing/centertap.
        resistance: Series resistance in ohms, stored as metadata only
        inductance: Inductance in henries, stored as metadata only
        add_pgs: When True, add a patterned ground shield on layers_pgs.
        pgs_diameter: Bounding size D of the ground shield square, in micrometers.
        pgs_width: Strip width w of each ground shield finger, in micrometers.
        pgs_spacing: Gap s between adjacent ground shield fingers, in micrometers.
        layer_winding: Metal layer for the main winding (top metal).
        layer_underpass: Metal layer for crossings and the center-tap bridge (one layer below layer_winding).
        layers_pgs: Layers on which the patterned ground shield is drawn.

    Returns:
        Component with 2 or 3 ports:
          P1  ->  left differential terminal   (layer_winding)
          P2  ->  right differential terminal  (layer_winding)
          CT  ->  center tap (only if center_tap=True), on layer_winding
                  if N <= 2, else on layer_underpass.
    """
    return cf.analog.inductors.symmetric_inductor(
        d_out=d_out,
        N=N,
        sides=sides,
        width=width,
        spacing=spacing,
        center_tap=center_tap,
        via_extent=via_extent,
        port_spacing=port_spacing,
        aspect_ratio=aspect_ratio,
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


if __name__ == "__main__":
    c = inductor()
    c.show()
