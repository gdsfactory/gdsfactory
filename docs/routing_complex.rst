Route to specific sides
======================================

.. autofunction:: pp.routing.route_ports_to_side.route_ports_to_side
.. autofunction:: pp.routing.get_bundle.link_ports
.. autofunction:: pp.routing.corner_bundle.corner_bundle
.. autofunction:: pp.routing.u_groove_bundle.u_bundle_direct
.. autofunction:: pp.routing.u_groove_bundle.u_bundle_indirect


 - `route_ports_to_side`, connect all the ports towards one bank of ports facing in one direction
 - `link_optical_ports`, banks or ports facing each other (but with arbitrary and varying pitch on each side)
 - `corner_bundle`, banks of ports with 90Deg / 270Deg between them (again pitch is flexible on both sides)
 - `u_bundle_direct`, banks of ports with direct U-turns
 - `u_bundle_indirect`, banks of ports with indirect U-turns

Each of these cases can either be directly. Another option is to call
`routing/get_bundle.py:get_bundle`

Get bundle acts as a high level entry point. Based on the angle configurations
of the banks of ports, it decides which sub-routine to call between:

 - `link_optical_ports`
 - `corner_bundle`
 - `u_bundle_direct`
 - `u_bundle_indirect`

For now it is not smart enough to decide whether it should call `route_ports_to_side`.
So you either need to connect your ports to face in one direction first, or to
use `route_ports_to_side` prior calling `get_bundle`

Example of `get_bundle` behavior when called with two banks of ports
(one list of input ports, another list of output ports). Nothing else is changed.
If different behaviors are required, several parameters can be used to tweak
the exact routes.
It is then recommended to explicitely call the wanted sub-routine with the wanted
arguments. e.g

`link_optical_ports`: `start_straight`, `end_straight`
`u_bundle_indirect`: `extension_length`

.. image:: images/connect_bundle.png


Routing banks of ports through pre-defined waypoints
-----------------------------------------------------

.. autofunction:: pp.routing.get_bundle_from_waypoints.get_bundle_from_waypoints

.. plot::
    :include-source:

    import numpy as np
    import pp


    @pp.cell
    def test_connect_bundle_waypoints():
        """Connect bundle of ports with bundle of routes following a list of waypoints."""
        xs1 = np.arange(10) * 5 - 500.0
        N = xs1.size
        ys2 = np.array([0, 5, 10, 20, 25, 30, 40, 55, 60, 75]) + 500.0

        ports1 = [pp.Port(f"A_{i}", (xs1[i], 0), 0.5, 90) for i in range(N)]
        ports2 = [pp.Port(f"B_{i}", (0, ys2[i]), 0.5, 180) for i in range(N)]

        c = pp.Component()
        way_points = [
            ports1[0].position,
            ports1[0].position + (0, 100),
            ports1[0].position + (200, 100),
            ports1[0].position + (200, -200),
            ports1[0].position + (0, -200),
            ports1[0].position + (0, -350),
            ports1[0].position + (400, -350),
            (ports1[0].x + 400, ports2[-1].y),
            ports2[-1].position,
        ]

        routes = pp.routing.get_bundle_from_waypoints(ports1, ports2, way_points)
        for route in routes:
            c.add(route['references'])

        return c

    cell = test_connect_bundle_waypoints()
    cell.plot()
