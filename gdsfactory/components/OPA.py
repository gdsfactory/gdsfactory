from functools import partial
from typing import Optional, Tuple, Union

import numpy as np

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.array_component import array as gf_array
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.ge_detector_straight_si_contacts import (
    ge_detector_straight_si_contacts,
)
from gdsfactory.components.pad import pad_array as pad_array_func
from gdsfactory.components.splitter_tree import splitter_tree
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.port import select_ports
from gdsfactory.types import ComponentSpec, CrossSectionSpec

default_det = partial(ge_detector_straight_si_contacts)


@cell
def OPA(
    bend: ComponentSpec = bend_euler,
    straight: ComponentSpec = straight_function,
    cross_section: CrossSectionSpec = "strip",
    grating_coupler: ComponentSpec = "grating_coupler_elliptical_te",
    num_couplers: Union[int, Tuple[int, int]] = 4,
    gc_spacing: Union[float, Tuple[float, float]] = 50.0,
    phase_shifter: ComponentSpec = "straight_pin",
    phase_shifter_length: float = 100.0,
    ps_num_cols: int = 1,
    ps_spacing: Union[float, Tuple[float, float]] = 50.0,
    gc_to_ps_spacing: float = 200.0,
    tap: ComponentSpec = "coupler",
    with_pads: bool = True,
    output_coupler: Optional[ComponentSpec] = None,
    photodetector: Optional[ComponentSpec] = default_det,
) -> Component:
    """Optical phased array with power combining.

    For now, it puts all of the phase shifters in a single column
    (ignores the ps_num_cols parameter)

    Args:
        bend: 90 degrees bend library.
        straight: straight function.
        cross_section: for routing (splitter to mzms and mzms to combiners).
        grating_coupler: grating coupler function.
        num_couplers: number of couplers in each direction. Can be a tuple with (num_x, num_y)
        gc_spacing: spacing between couplers [um]. Can be a tuple with (dx, dy).
        phase_shifter: phase_shifter function.
        phase_shifter_length: length of the phase shifter
        ps_num_cols: number of columns in which to distribute the phase shifters.
        ps_spacing: spacing between phase shifters [um]. Can be a tuple with (dx, dy).
        gc_to_ps_spacing: spacing between the GC array and the phase shifter array.
        combiner: combiner function.
        tap: function for the tap for feedback signal
        with_pads: if True, we draw pads for the photodetector and the negative electrode of all ps.
        output_coupler: Optional coupler to add after the combiner
        photodetector: Optional photodetector to connect to the tap output.
    """
    c = Component()

    bend_comp = gf.get_component(bend)
    bend_dx = 2 * bend_comp.xmax - bend_comp.xmin
    bend_dy = 2 * bend_comp.ymax - bend_comp.ymin

    if not isinstance(gc_spacing, tuple):
        gc_spacing = (gc_spacing, gc_spacing)

    if not isinstance(num_couplers, tuple):
        num_couplers = (num_couplers, num_couplers)

    if not isinstance(ps_spacing, tuple):
        ps_spacing = (ps_spacing, ps_spacing)

    num_elems = num_couplers[0] * num_couplers[1]
    nx = num_couplers[0]
    ny = num_couplers[1]

    ps_num_rows = int(np.ceil(num_elems / ps_num_cols))

    # ------ First of, draw the grating couplers and
    # phase shifters with the right spacing ---------

    gc = gf.get_component(grating_coupler)
    gc_array = gf_array(component=gc, spacing=gc_spacing, columns=nx, rows=ny)

    gc_ar = c << gc_array
    gc_ar.mirror((0, 1))

    # Important for routing afterwards
    real_gc_spacing = (
        gc_spacing[0] - np.abs(gc.xmax) - np.abs(gc.xmin),
        gc_spacing[1] - np.abs(gc.ymax) - np.abs(gc.ymin),
    )

    ps = gf.get_component(phase_shifter, length=phase_shifter_length)
    ps_array = gf_array(
        component=ps, spacing=ps_spacing, columns=ps_num_cols, rows=ps_num_rows
    )

    ps_ar = c << ps_array

    ps_ar.xmin = gc_ar.xmax + gc_to_ps_spacing
    ps_ar.y = gc_ar.y + gc_spacing[1] / 2 - ps_spacing[1] / 2

    # ---- Now do the combining of powers ----

    tree = splitter_tree(
        noutputs=num_elems,
        spacing=[40.0, 2 * ps_spacing[1]],
        cross_section=cross_section,
    )
    sp_tree = c << tree
    sp_tree.mirror((0, 1))
    sp_tree.xmin = ps_ar.xmax + 20.0
    sp_tree.y = ps_ar.y

    # Connect the splitter tree to the phase shifter (should be easy since they are straight)
    ps_ports = ps_ar.get_ports_dict(port_type="optical", prefix="o2")
    sp_ports = sp_tree.get_ports_dict(port_type="optical", prefix="o2")
    routes = gf.routing.get_bundle(
        ps_ports,
        sp_ports,
        sort_ports=True,
        cross_section=cross_section,
        bend=bend,
        straight=straight,
    )
    for route in routes:
        c.add(route.references)

    # ---- Add the tap ----
    tap = gf.get_component(tap)
    tp = c << tap

    tp.connect("o1", sp_tree.ports["o1_0_0"])

    # --- Output coupler if present ---

    if output_coupler:
        # Add output coupler
        output_coupler = gf.get_component(output_coupler)
        out_coup = c << output_coupler
        out_coup.connect("o1", tp.ports["o3"])
    else:
        # Create a port
        c.add_port("signal_out", port=tp.ports["o3"])

    # --- PD for the tap if present --

    if photodetector:
        # Add photodetector
        photodetector = gf.get_component(photodetector)
        if with_pads:
            photodetector = gf.routing.add_electrical_pads_top(
                component=photodetector,
                direction="right",
                spacing=(100, -136),
                select_ports=partial(
                    select_ports, names=["top_e3", "bot_e3"], clockwise=False
                ),
                pad_array=partial(pad_array_func, columns=1, rows=1),
            )
        pd = c << photodetector
        pd.connect("o1", tp.ports["o4"])
    else:
        # Create a port
        c.add_port("tap_out", port=tp.ports["o4"])

    # --- Now for the hard part - connect GCs to PS ----

    # The approach is relatively simple - each row has routing on top of the GCs row
    for row in range(1, ny + 1):

        # Get the relevant ports
        gc_ports_row = gc_ar.get_ports_list(
            port_type="optical", prefix="o1_" + str(row)
        )

        ps_names = []
        for i in range((row - 1) * nx + 1, row * nx + 1):
            ps_names.append("o1_" + str(i) + "_1")
        ps_ports_row = ps_ar.get_ports_list(port_type="optical", names=ps_names)

        # We need to manually make the routes
        if row > int(np.floor(ny / 2)):
            x_step = np.floor(ny / 2) - np.mod(row - 1, int(np.floor(ny / 2)))
            sign = -1
        else:
            x_step = np.mod(row - 1, int(np.floor(ny / 2))) + 1
            sign = 1

        x_spacing = 5.0
        col_num = 1
        half_nx = int(np.ceil(nx / 2) + 1)
        for gc_port, ps_port in zip(gc_ports_row, ps_ports_row):

            y_step = np.mod(col_num - 1, nx)

            p0 = gc_port.center
            p1 = ps_port.center

            if y_step > 0:
                dy1 = np.max(
                    [
                        y_step * real_gc_spacing[1] / nx + gc.ymax * (col_num > 1),
                        bend_dy,
                    ]
                )
            else:
                dy1 = 0
            dx1 = np.max([0.5 * bend_dx, 0.8 * real_gc_spacing[0]])
            x2 = (
                gc_ar.xmax
                + gc_to_ps_spacing * x_step / half_nx
                - sign * x_spacing * (nx / 2 - col_num)
            )
            routes = gf.routing.get_route_from_waypoints(
                waypoints=[
                    p0,
                    p0 + (dx1, 0),
                    p0 + (dx1, dy1),
                    (x2, p0[1] + dy1),
                    (x2, p1[1]),
                    p1,
                ],
                straight=straight,
                bend=bend,
                cross_section=cross_section,
            )
            c.add(routes.references)
            col_num += 1

    # --- Pads if necessary, else raise the electrical ports
    if with_pads:
        # Connect only all negative electrodes together
        neg_pad = c << gf.components.pad(port_orientation=0)
        neg_pad.xmax = ps_ar.xmin - 50.0
        neg_pad.ymin = ps_ar.ymax + 20.0

        ps_neg_el_ports = ps_ar.get_ports_list(port_type="electrical", prefix="top_e1")
        for port in ps_neg_el_ports:
            route = gf.routing.get_route_electrical(port, neg_pad.ports["e4"])
            c.add(route.references)

        # Raise the positive electrodes
        c.add_ports(
            ps_ar.get_ports_list(port_type="electrical", prefix="bot"), prefix="psarray"
        )

    else:
        c.add_ports(ps_ar.get_ports_list(port_type="electrical"), prefix="psarray")
        if photodetector:
            c.add_ports(pd.get_ports_list(port_type="electrical"), prefix="pd")
        c.auto_rename_ports()

    return c
