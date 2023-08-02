"""
In a multiple-topology Clements Scheme we can implement any universal photonic function.
"""
from __future__ import annotations

import numpy as np

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.mzi import mzi2x2_2x2
from gdsfactory.components.straight import straight
from gdsfactory.port import select_ports_electrical
from gdsfactory.routing import get_route


def find_largest_component(component_list: list) -> Component:
    # TODO optimise speed here
    largest_component = component_list[0]
    for component_i in component_list:
        # If element is larger than largest component
        if (component_i.xsize * component_i.ysize) > (
            largest_component.xsize * largest_component.ysize
        ):
            largest_component = component_i
    return largest_component


@cell
def component_lattice_generic(
    network: list[list] | None = None,
) -> Component:
    """
    The shape of the `network` matrix determines the physical interconnection.
    Note that there should be at least S+1=N modes
    based on this formalism of interconnection,
    and the position of the component implements a connectivity in between the modes,
    and assumes a 2x2 network encoding.
    One nice functionality by this component is that it can generate a
    component lattice for generic variable components with different x and y pitches.
    Initially this will maximise the surface area required
    but different placement algorithms can compact the size.

    Args:
        network: A list of lists of components that are to be placed in the lattice.

    Returns:
        Component: A component lattice that implements the physical network.

    The placement matrix is in this form:
    .. math::

        M = X & 0 & X
            0 & P & 0
            X & 0 & X


    :include-source:
        import gdsfactory as gf
        from gdsfactory.components.mzi import mzi2x2_2x2

        example_component_lattice = [
            [mzi2x2_2x2(), 0, mzi2x2_2x2()],
            [0, mzi2x2_2x2(delta_length=30.0), 0],
            [mzi2x2_2x2(), 0, mzi2x2_2x2()],
        ]
        c = gf.components.component_lattice_generic(example_component_lattice)


    Another example that demonstrates the generic-nature of this component lattice
    algorithm can be with an mixed set of actively driven and passiver interferometers.
    The placement matrix is in this form:

    .. math::

        M = Y & 0 & A
            0 & B & 0
            C & 0 & Y

    :include-source:
        import gdsfactory as gf
        from gdsfactory.components import mzi2x2_2x2_phase_shifter, mzi2x2_2x2

        example_mixed_component_lattice = [
            [mzi2x2_2x2_phase_shifter(), 0, mzi2x2_2x2(delta_length=20.0)],
            [0, mzi2x2_2x2(delta_length=30.0), 0],
            [mzi2x2_2x2(delta_length=15.0), 0, mzi2x2_2x2_phase_shifter()],
        ]
        c = gf.components.component_lattice_generic(
            network=example_mixed_component_lattice
        )

    # TODO implement balanced waveguide paths function per stage
    # TODO automatic electrical fanout?
    # TODO multiple placement optimization algorithms.
    """

    network = network or [
        [mzi2x2_2x2(), 0, mzi2x2_2x2()],
        [0, mzi2x2_2x2(delta_length=30.0), 0],
        [mzi2x2_2x2(), 0, mzi2x2_2x2()],
    ]

    element_references = list()
    network = np.array(network)
    # Check number of dimensions is 2
    if network.ndim != 2:
        # Get the length and then width of the array
        raise AttributeError(
            "Physical network dimensions don't work."
            "Check the dimensional structure of your network matrix."
        )

    C = Component()
    # Estimate the size of the network fabric
    elements_list = np.vstack([np.nonzero(network), network[np.nonzero(network)]])
    largest_component = find_largest_component(elements_list[2])  # List of elements
    component_column_amount = len(network[0])
    mode_amount = component_column_amount + 1
    inter_stage_clearance_x_offset = 40
    inter_stage_clearance_y_offset = 40
    x_length = (
        mode_amount * largest_component.xsize
        + mode_amount * inter_stage_clearance_x_offset
    )
    y_length = (
        mode_amount * largest_component.ysize
        + mode_amount * inter_stage_clearance_y_offset
    )
    x_component_pitch = x_length / mode_amount
    y_component_pitch = y_length / mode_amount
    x_mode_pitch = x_length / mode_amount
    y_mode_pitch = y_length / mode_amount
    # each distinct operation on the network is a separate iteration for-loop so new
    # functionality can be extended and easily identified.

    # Create all the waveguides inputs and outputs
    # Originally implemented in gdsfactory array but got netlist errors
    interconnection_ports_array = []
    for column_j in range(mode_amount):
        interconnection_ports_array.append([])
        for row_i in range(mode_amount):
            straight_i = C << straight(length=1, width=0.5)
            interconnection_ports_array[column_j].extend([straight_i])
            interconnection_ports_array[column_j][row_i].move(
                destination=(x_mode_pitch * column_j, -y_mode_pitch * row_i)
            )

            if column_j == 0:
                # Inputs
                C.add_port(
                    port=straight_i.ports["o1"],
                    name="in_o_" + str(row_i),
                )
            elif column_j == (mode_amount - 1):
                # Outputs
                C.add_port(
                    port=straight_i.ports["o2"],
                    name="out_o_" + str(row_i),
                )

    # Place the components in between the corresponding waveguide modes
    j = 0
    k = 0
    for column_j in network:
        i = 0
        for element_i in column_j:
            # Check if element is nonzero
            if element_i != 0:
                element_references.append(C << element_i)
                element_references[k].center = (0, 0)
                element_references[k].move(
                    destination=(
                        x_component_pitch * j
                        + largest_component.xsize / 2
                        + inter_stage_clearance_x_offset / 2,
                        -y_component_pitch * i - inter_stage_clearance_y_offset,
                    )
                )
                k += 1
            i += 1
        j += 1

    # Go position by position to place and connect everything, column by column
    j = 0
    k = 0
    for column_j in network:
        i = 0
        # Connect the modes together
        for element_i in column_j:
            # Row in column
            if element_i != 0:
                # Connect the adjacent input waveguide ports to the first element columns
                # if j == 0:
                route_0 = get_route(
                    interconnection_ports_array[j][i].ports["o2"],
                    element_references[k].ports["o2"],
                    radius=5,
                )
                route_i = get_route(
                    interconnection_ports_array[j][i + 1].ports["o2"],
                    element_references[k].ports["o1"],
                    radius=5,
                )
                # Connect output of the component to the component
                route_0_out = get_route(
                    interconnection_ports_array[j + 1][i].ports["o1"],
                    element_references[k].ports["o3"],
                    radius=5,
                )
                route_i_out = get_route(
                    interconnection_ports_array[j + 1][i + 1].ports["o1"],
                    element_references[k].ports["o4"],
                    radius=5,
                )
                C.add(route_0.references)
                C.add(route_i.references)
                C.add(route_0_out.references)
                C.add(route_i_out.references)
                k += 1

            elif element_i == 0:
                # When no element at junction, connect straight ahead between
                if i == 0:
                    # If at start top row then just connect top
                    route_i = get_route(
                        interconnection_ports_array[j][i].ports["o2"],
                        interconnection_ports_array[j + 1][i].ports["o1"],
                        radius=5,
                    )
                    C.add(route_i.references)
                elif i == (len(column_j) - 1):
                    # If at end then connect bottom
                    route_i = get_route(
                        interconnection_ports_array[j][i + 1].ports["o2"],
                        interconnection_ports_array[j + 1][i + 1].ports["o1"],
                        radius=5,
                    )
                    C.add(route_i.references)

                    if column_j[i - 1] != 0:
                        # If previous element nonzero then pass
                        pass
                    elif column_j[i - 1] == 0:
                        # If previous element is zero then connect top straight
                        route_i = get_route(
                            interconnection_ports_array[j][i].ports["o2"],
                            interconnection_ports_array[j + 1][i].ports["o1"],
                            radius=5,
                        )
                        C.add(route_i.references)

                elif column_j[i - 1] == 0:
                    # If previous element is zero then connect top straight
                    route_i = get_route(
                        interconnection_ports_array[j][i].ports["o2"],
                        interconnection_ports_array[j + 1][i].ports["o1"],
                        radius=5,
                    )
                    C.add(route_i.references)

                elif column_j[i - 1] != 0:
                    # If previous element nonzero then pass
                    pass
            i += 1
        j += 1

    # Append electrical ports to component to total connectivity can be constructed.
    j = 0
    k = 0
    for column_j in network:
        i = 0
        for element_i in column_j:
            # Check if element is nonzero
            if element_i != 0:
                electrical_ports_list_i = select_ports_electrical(
                    element_references[k].ports
                ).items()
                if len(electrical_ports_list_i) > 0:
                    # Electrical ports exist in component
                    for electrical_port_i in electrical_ports_list_i:
                        C.add_port(
                            port=electrical_port_i[1],
                            name=electrical_port_i[0] + "_" + str(i) + "_" + str(j),
                        )
                        # Row column notation
                k += 1
            i += 1
        j += 1
    return C


if __name__ == "__main__":
    # from gdsfactory.components.mzi import mzi2x2_2x2
    # from gdsfactory.components.mzi_phase_shifter import mzi2x2_2x2_phase_shifter

    # example_component_lattice = [
    #     [mzi2x2_2x2(), 0, mzi2x2_2x2()],
    #     [0, mzi2x2_2x2(), 0],
    #     [mzi2x2_2x2(), 0, mzi2x2_2x2()],
    # ]
    # c = component_lattice_generic(example_component_lattice)
    # c.show(show_ports=True)

    # example_mixed_component_lattice = [
    #     [mzi2x2_2x2_phase_shifter(), 0, mzi2x2_2x2(delta_length=20.0)],
    #     [0, mzi2x2_2x2(delta_length=50.0), 0],
    #     [mzi2x2_2x2(delta_length=100.0), 0, mzi2x2_2x2_phase_shifter()],
    # ]
    # c_mixed = component_lattice_generic(example_mixed_component_lattice)
    # c_mixed.show(show_ports=True)
    c = component_lattice_generic()
    c.show()
