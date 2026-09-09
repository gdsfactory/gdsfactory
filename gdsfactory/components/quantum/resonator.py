from __future__ import annotations

__all__ = ["resonator_cpw", "resonator_lumped", "resonator_quarter_wave"]

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.cross_section import CrossSection, Section
from gdsfactory.port import Port
from gdsfactory.typings import LayerSpec


def _add_meander(
    c: Component,
    run: float,
    radius: float,
    turns: int,
    cross_section: CrossSection,
    port: Port | None = None,
) -> tuple[Port, Port]:
    """Adds a serpentine of straights and 180 degree bends to ``c``.

    The serpentine starts heading along +x. Every 180 degree bend flips the
    heading and steps down by ``2 * radius``, so consecutive runs are stacked
    with a pitch of ``2 * radius``.

    Args:
        c: component to add the serpentine to.
        run: length of each straight run in um.
        radius: bend radius of the 180 degree turns in um.
        turns: number of 180 degree turns. ``turns + 1`` straight runs are drawn.
        cross_section: cross section to extrude.
        port: optional port to connect the first straight run to.

    Returns:
        The input and output ports of the serpentine.
    """
    if run <= 0:
        raise ValueError(f"{run=} must be positive")
    if radius <= 0:
        raise ValueError(f"{radius=} must be positive")
    if turns < 0:
        raise ValueError(f"{turns=} must not be negative")

    straight = gf.components.straight(length=run, cross_section=cross_section)
    bends = [
        gf.components.bend_circular(
            angle=angle, radius=radius, cross_section=cross_section
        )
        for angle in (-180.0, 180.0)
    ]

    previous = c << straight
    if port is not None:
        previous.connect("o1", port)
    port_in = previous.ports["o1"]
    for i in range(turns):
        bend = c << bends[i % 2]
        bend.connect("o1", previous.ports["o2"])
        previous = c << straight
        previous.connect("o1", bend.ports["o2"])
    return port_in, previous.ports["o2"]


def _meander_length(run: float, radius: float, turns: int) -> float:
    """Returns the path length of a serpentine drawn by :func:`_add_meander`."""
    return (turns + 1) * run + turns * np.pi * radius


def _cpw_cross_section(
    width: float,
    gap: float,
    radius: float,
    layer_metal: LayerSpec,
    layer_gap: LayerSpec,
    port_type: str,
) -> CrossSection:
    """Returns a coplanar waveguide cross section.

    The center conductor is drawn on ``layer_metal`` and the two slots etched
    out of the surrounding ground plane are drawn on ``layer_gap``.

    Args:
        width: width of the center conductor in um.
        gap: width of each slot in um.
        radius: default bend radius in um.
        layer_metal: layer for the center conductor.
        layer_gap: layer for the two slots.
        port_type: port type for the two ends of the extrusion.
    """
    offset = (width + gap) / 2
    return CrossSection(
        sections=(
            Section(
                width=width,
                layer=layer_metal,
                port_names=("o1", "o2"),
                port_types=(port_type, port_type),
                name="center",
            ),
            Section(width=gap, offset=offset, layer=layer_gap, name="slot_top"),
            Section(width=gap, offset=-offset, layer=layer_gap, name="slot_bot"),
        ),
        radius=radius,
    )


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
    if meander_width <= meander_pitch:
        raise ValueError(f"{meander_width=} must be larger than {meander_pitch=}")
    if length <= 0:
        raise ValueError(f"{length=} must be positive")

    c = Component()

    radius = meander_pitch / 2
    run = meander_width - meander_pitch
    bend_length = np.pi * radius

    # Solve length = (turns + 1) * run + turns * bend_length for the turn count.
    turns = max(0, round((length - run) / (run + bend_length)))

    xs_main = _cpw_cross_section(
        width=width,
        gap=gap,
        radius=radius,
        layer_metal=layer_metal,
        layer_gap=layer_gap,
        port_type=port_type,
    )
    port_in, port_out = _add_meander(
        c, run=run, radius=radius, turns=turns, cross_section=xs_main
    )

    if coupling_length > 0:
        xs_coupling = _cpw_cross_section(
            width=width,
            gap=coupling_gap,
            radius=radius,
            layer_metal=layer_metal,
            layer_gap=layer_gap,
            port_type=port_type,
        )
        arm = gf.components.straight(length=coupling_length, cross_section=xs_coupling)
        arm_in = c << arm
        arm_out = c << arm
        arm_in.connect("o1", port_in)
        arm_out.connect("o1", port_out)
        port_in = arm_in.ports["o2"]
        port_out = arm_out.ports["o2"]

    c.add_port(name="input", port=port_in)
    c.add_port(name="output", port=port_out)

    actual_length = (
        _meander_length(run=run, radius=radius, turns=turns) + 2 * coupling_length
    )

    # Add metadata
    c.info["resonator_type"] = "cpw"
    c.info["length"] = actual_length
    c.info["width"] = width
    c.info["gap"] = gap
    c.info["meander_turns"] = turns
    c.info["frequency_estimate"] = (
        3e8 / (2 * actual_length * 1e-6) / 1e9
    )  # GHz in vacuum, rough estimate

    if port_type == "electrical":
        for p in list(c.ports):
            if p.name and p.port_type == "electrical":
                c.create_pin(ports=[p], name=p.name)

    c.flatten()
    return c


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
    if inductor_turns < 0:
        raise ValueError(f"{inductor_turns=} must not be negative")

    c = Component()

    xs = CrossSection(
        sections=(
            Section(
                width=inductor_width,
                layer=layer_metal,
                port_names=("o1", "o2"),
                port_types=(port_type, port_type),
                name="wire",
            ),
        ),
        radius=inductor_radius,
    )

    capacitor = gf.get_component(
        "interdigital_capacitor",
        fingers=capacitor_fingers,
        finger_length=capacitor_finger_length,
        finger_gap=capacitor_finger_gap,
        thickness=capacitor_thickness,
        layer=layer_metal,
    )
    cap_ref = c << capacitor

    # Straight interconnect from the capacitor to the inductor.
    link = c << gf.components.straight(length=coupling_gap, cross_section=xs)
    link.connect(
        "o1",
        cap_ref.ports["o2"],
        allow_width_mismatch=True,
        allow_type_mismatch=True,
    )

    _, ind_out = _add_meander(
        c,
        run=inductor_run,
        radius=inductor_radius,
        turns=inductor_turns,
        cross_section=xs,
        port=link.ports["o2"],
    )

    c.add_port(name="input", port=cap_ref.ports["o1"], port_type=port_type)
    c.add_port(name="output", port=ind_out)

    # Add metadata
    c.info["resonator_type"] = "lumped"
    c.info["capacitor_fingers"] = capacitor_fingers
    c.info["inductor_turns"] = inductor_turns
    c.info["inductor_radius"] = inductor_radius
    c.info["inductor_length"] = _meander_length(
        run=inductor_run, radius=inductor_radius, turns=inductor_turns
    )

    if port_type == "electrical":
        for p in list(c.ports):
            if p.name and p.port_type == "electrical":
                c.create_pin(ports=[p], name=p.name)

    return c


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
    c = Component()

    # Create main resonator line
    main_line = gf.components.rectangle(
        size=(length, width),
        layer=layer_metal,
    )
    c.add_ref(main_line)

    # Create shorting stub at one end
    short_stub = gf.components.rectangle(
        size=(short_stub_length, width + 2 * gap),
        layer=layer_metal,
    )
    short_ref = c.add_ref(short_stub)
    short_ref.move((length, -gap))

    # Create ground planes
    ground_top = gf.components.rectangle(
        size=(length + short_stub_length + 2 * gap, gap),
        layer=layer_metal,
    )
    ground_top_ref = c.add_ref(ground_top)
    ground_top_ref.move((-gap, width))

    ground_bottom = gf.components.rectangle(
        size=(length + short_stub_length + 2 * gap, gap),
        layer=layer_metal,
    )
    ground_bottom_ref = c.add_ref(ground_bottom)
    ground_bottom_ref.move((-gap, -gap))

    # Create coupling region at open end
    coupling_region = gf.components.rectangle(
        size=(coupling_length, coupling_gap),
        layer=layer_gap,
    )
    coupling_ref = c.add_ref(coupling_region)
    coupling_ref.move((-coupling_length, width / 2 - coupling_gap / 2))

    # Add port for coupling
    c.add_port(
        name="coupling",
        center=(-coupling_length / 2, width / 2),
        width=coupling_gap,
        orientation=180,
        layer=layer_metal,
        port_type=port_type,
    )

    # Add metadata
    c.info["resonator_type"] = "quarter_wave"
    c.info["length"] = length
    c.info["width"] = width
    c.info["gap"] = gap
    c.info["frequency_estimate"] = (
        3e8 / (4 * length * 1e-6) / 1e9
    )  # GHz, rough estimate

    if port_type == "electrical":
        for p in list(c.ports):
            if p.name and p.port_type == "electrical":
                c.create_pin(ports=[p], name=p.name)

    return c
