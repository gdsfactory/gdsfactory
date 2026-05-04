"""1x16 optical switch tree using cascaded MZI 2x2 switches."""

__all__ = ["switch_nxn", "switch_nxn_with_fiber_array"]

from functools import partial

import gdsfactory as gf
from doroutes.multilayer import (
    MetalLayerSpec,
    RouteNetSpec,
    RoutingConfig,
    route_nets_deterministic_copy,
)
from doroutes.multilayer.engine_multinet import _precompute_port_geometries
from gdsfactory.components.containers.splitter_tree import splitter_tree
from gdsfactory.components.mzis import mzi1x2_2x2
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Spacing


@gf.cell
def via_m2_m3(width: float = 11.0, length: float = 11.0) -> gf.Component:
    """Via transition between M2 and M3 for DoRoutes."""
    return gf.components.via_stack(
        size=(width, length), layers=("M2", "MTOP"), vias=("via2", None)
    )


GPDK_CONFIG = RoutingConfig(
    metal_layers=(
        MetalLayerSpec(
            layer_tuple=(45, 0),  # M2
            preferred_direction="h",
            min_width=10.0,
            min_via_pad=11.0,
            below_cut_layer=None,
            pitch=20.0,
            spacing=5.0,
        ),
        MetalLayerSpec(
            layer_tuple=(49, 0),  # M3
            preferred_direction="v",
            min_width=10.0,
            min_via_pad=11.0,
            below_cut_layer=(43, 0),  # VIA2
            pitch=20.0,
            spacing=5.0,
        ),
    ),
    via_factory=via_m2_m3,
    via_metal_enclosure_add=0.5,
)

_mzi1x2_2x2 = partial(
    mzi1x2_2x2,
    combiner="mmi2x2",
    delta_length=0,
    straight_x_top="straight_heater_metal",
    length_x=None,
)


@gf.cell
def switch_nxn(
    coupler: ComponentSpec = _mzi1x2_2x2,
    spacing: Spacing = (500, 100),
    bend_s: ComponentSpec | None = "bend_s",
    bend_s_xsize: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    noutputs: int = 16,
) -> gf.Component:
    """1x16 optical switch tree using cascaded MZI 2x2 switches.

    Args:
        coupler: coupler factory (default: MZI 1x2 to 2x2 switch element).
        spacing: x, y spacing between couplers.
        bend_s: S-bend function for termination.
        bend_s_xsize: xsize for the S-bend.
        cross_section: cross_section spec.
        noutputs: number of outputs.
    """
    return splitter_tree(
        coupler=coupler,
        noutputs=noutputs,
        spacing=spacing,
        bend_s=bend_s,
        bend_s_xsize=bend_s_xsize,
        cross_section=cross_section,
    )


@gf.cell
def switch_nxn_with_fiber_array(
    coupler: ComponentSpec = _mzi1x2_2x2,
    spacing: Spacing = (500, 100),
    bend_s: ComponentSpec | None = "bend_s",
    bend_s_xsize: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    pad: ComponentSpec = "pad",
    pad_pitch: float = 120.0,
    cross_section_metal: CrossSectionSpec = "metal_routing",
    noutputs: int = 2**2,
) -> gf.Component:
    """1x16 switch tree with fiber array grating couplers and electrical pads.

    Args:
        coupler: coupler factory (default: MZI 1x2 to 2x2 switch element).
        spacing: x, y spacing between couplers.
        bend_s: S-bend function for termination.
        bend_s_xsize: xsize for the S-bend.
        cross_section: cross_section spec.
        pad: electrical pad spec.
        pad_pitch: pitch between electrical pads.
        cross_section_metal: metal cross section for electrical routing.
        noutputs: number of outputs.
    """
    c = gf.Component()

    switch = c << switch_nxn(
        coupler=coupler,
        spacing=spacing,
        bend_s=bend_s,
        bend_s_xsize=bend_s_xsize,
        cross_section=cross_section,
        noutputs=noutputs,
    )
    c.add_ports(switch.ports.filter(port_type="optical"))

    # Collect and add electrical ports for heater pads
    electrical_ports = list(switch.ports.filter(port_type="electrical", orientation=90))
    electrical_ports.sort(key=lambda p: p.center[0])

    if electrical_ports:
        npads = len(electrical_ports)
        pads = c << gf.components.array(
            component=pad,
            columns=npads,
            column_pitch=pad_pitch,
        )
        pads.x = switch.x
        pads.ymin = switch.ymax + 900

        pad_ports = list(pads.ports.filter(orientation=270))[:npads]
        pad_ports.sort(key=lambda p: p.center[0])

        # Add routing area polygon so the A* grid covers the full region
        routing_area_layer = (235, 4)
        margin = 50.0
        all_pts = [p.center for p in electrical_ports] + [p.center for p in pad_ports]
        xs = [pt[0] for pt in all_pts]
        ys = [pt[1] for pt in all_pts]
        c.add_polygon(
            [
                (min(xs) - margin, min(ys) - margin),
                (max(xs) + margin, min(ys) - margin),
                (max(xs) + margin, max(ys) + margin),
                (min(xs) - margin, max(ys) + margin),
            ],
            layer=routing_area_layer,
        )

        # Build per-wire RouteNetSpecs
        nets = [
            RouteNetSpec(
                name=f"e{i}",
                start=electrical_ports[i],
                stop=pad_ports[i],
                port_name_prefix=f"e{i}",
            )
            for i in range(min(npads, len(electrical_ports)))
        ]

        layers_to_avoid = [(45, 0), (49, 0)]

        # Use the smaller of the two port polygons' longest length as width
        geom_cache = _precompute_port_geometries(c, nets, GPDK_CONFIG, width=1.0)
        dbu = c.kcl.dbu
        net_widths = []
        for net in nets:
            for _label, port in [("start", net.start), ("stop", net.stop)]:
                key = (int(port.dcenter[0] / dbu), int(port.dcenter[1] / dbu))
                geom = geom_cache.get(key)
                if geom:
                    longest = max(geom.orig_extent_x_um, geom.orig_extent_y_um)
                    net_widths.append(longest)
        # Per-net: min of start/stop longest; global: min across all nets
        per_net = [
            min(net_widths[i * 2], net_widths[i * 2 + 1]) for i in range(len(nets))
        ]
        width = min(per_net) if per_net else 10.0

        c, _ = route_nets_deterministic_copy(
            c,
            nets=nets,
            config=GPDK_CONFIG,
            grid_unit=10.0,
            width=width,
            dynamic_width=False,
            layers_to_avoid=layers_to_avoid,
            clearance=5.0,
            deterministic=True,
            require_all=False,
            geom_cache=geom_cache,
        )

        # Remove routing area marker
        c.remove_layers(layers=[routing_area_layer])

        for i, port in enumerate(pad_ports):
            c.add_port(name=f"e{i}", port=port)

    return c
