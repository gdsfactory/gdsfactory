from __future__ import annotations

__all__ = ["resonator_cpw", "resonator_lumped", "resonator_quarter_wave"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["quantum"])
def resonator_cpw(
    length: float = 1000.0,
    width: float = 10.0,
    gap: float = 6.0,
    meander_pitch: float = 50.0,
    meander_width: float = 200.0,
    coupling_gap: float = 5.0,
    coupling_length: float = 100.0,
    layer_metal: LayerSpec = (1, 0),
    layer_gap: LayerSpec = (2, 0),
    port_type: str = "electrical",
) -> Component:
    """Creates a half-wave coplanar waveguide (CPW) resonator.

    The resonator is a meandered coplanar waveguide: a center conductor on
    ``layer_metal`` flanked by two slots on ``layer_gap`` which are etched out of
    the surrounding ground plane. Both ends are straight coupling arms whose
    slots are widened (or narrowed) to ``coupling_gap`` so the coupling to a
    feedline or qubit can be tuned independently of the resonator impedance.

    Args:
        length: Target length of the meandered section in μm. The number of
            meander turns is rounded to fit; the length actually drawn is
            reported in ``component.info["length"]``.
        width: Width of the center conductor in μm.
        gap: Slot width on each side of the center conductor in μm.
        meander_pitch: Pitch between meander runs in μm. Sets the bend radius to
            ``meander_pitch / 2``.
        meander_width: Span of each meander run in μm. Must exceed
            ``meander_pitch``.
        coupling_gap: Slot width in the coupling arms in μm.
        coupling_length: Length of each coupling arm in μm.
        layer_metal: Layer for the metal center conductor.
        layer_gap: Layer for the slots etched out of the ground plane.
        port_type: Type of port to add to the component.

    Returns:
        Component: A gdsfactory component with the CPW resonator geometry.
    """
    return cf.resonator_cpw(
        length=length,
        width=width,
        gap=gap,
        meander_pitch=meander_pitch,
        meander_width=meander_width,
        coupling_gap=coupling_gap,
        coupling_length=coupling_length,
        layer_metal=layer_metal,
        layer_gap=layer_gap,
        port_type=port_type,
    )


@gf.cell_with_module_name(tags=["quantum"])
def resonator_lumped(
    capacitor_fingers: int = 4,
    capacitor_finger_length: float = 20.0,
    capacitor_finger_gap: float = 2.0,
    capacitor_thickness: float = 5.0,
    inductor_width: float = 2.0,
    inductor_turns: int = 3,
    inductor_radius: float = 20.0,
    inductor_run: float = 60.0,
    coupling_gap: float = 5.0,
    layer_metal: LayerSpec = (1, 0),
    port_type: str = "electrical",
) -> Component:
    """Creates a lumped element resonator: an interdigital capacitor in series with a meander inductor.

    The interdigital capacitor and the meander inductor are galvanically
    connected by a straight interconnect, so the component is a two terminal
    series LC with ports on the outer terminal of each element.

    Args:
        capacitor_fingers: Number of fingers in the interdigital capacitor.
        capacitor_finger_length: Length of each capacitor finger in μm.
        capacitor_finger_gap: Gap between capacitor fingers in μm.
        capacitor_thickness: Thickness of capacitor fingers in μm.
        inductor_width: Width of the inductor wire in μm.
        inductor_turns: Number of 180 degree turns in the meander inductor.
        inductor_radius: Bend radius of the meander inductor in μm.
        inductor_run: Length of each straight run of the meander inductor in μm.
        coupling_gap: Length of the interconnect between capacitor and inductor in μm.
        layer_metal: Layer for the metal structures.
        port_type: Type of port to add to the component.

    Returns:
        Component: A gdsfactory component with the lumped resonator geometry.
    """
    return cf.resonator_lumped(
        capacitor_fingers=capacitor_fingers,
        capacitor_finger_length=capacitor_finger_length,
        capacitor_finger_gap=capacitor_finger_gap,
        capacitor_thickness=capacitor_thickness,
        inductor_width=inductor_width,
        inductor_turns=inductor_turns,
        inductor_radius=inductor_radius,
        inductor_run=inductor_run,
        coupling_gap=coupling_gap,
        layer_metal=layer_metal,
        port_type=port_type,
    )


@gf.cell_with_module_name(tags=["quantum"])
def resonator_quarter_wave(
    length: float = 2500.0,
    width: float = 10.0,
    gap: float = 6.0,
    short_stub_length: float = 50.0,
    coupling_gap: float = 5.0,
    coupling_length: float = 100.0,
    layer_metal: LayerSpec = (1, 0),
    layer_gap: LayerSpec = (2, 0),
    port_type: str = "electrical",
) -> Component:
    """Creates a quarter-wave coplanar waveguide resonator.

    A quarter-wave resonator is shorted at one end and has maximum electric field
    at the open end, making it suitable for capacitive coupling.

    Args:
        length: Length of the quarter-wave resonator in μm.
        width: Width of the center conductor in μm.
        gap: Gap width on each side of the center conductor in μm.
        short_stub_length: Length of the shorting stub in μm.
        coupling_gap: Gap for capacitive coupling in μm.
        coupling_length: Length of the coupling region in μm.
        layer_metal: Layer for the metal conductor.
        layer_gap: Layer for the gaps.
        port_type: Type of port to add to the component.

    Returns:
        Component: A gdsfactory component with the quarter-wave resonator geometry.
    """
    return cf.resonator_quarter_wave(
        length=length,
        width=width,
        gap=gap,
        short_stub_length=short_stub_length,
        coupling_gap=coupling_gap,
        coupling_length=coupling_length,
        layer_metal=layer_metal,
        layer_gap=layer_gap,
        port_type=port_type,
    )
