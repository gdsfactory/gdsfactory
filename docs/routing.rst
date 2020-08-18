Routing
=============================


Connecting ports
----------------------

Connections are made along manhattan routes using the `routing/manhattan.py` module.
Convenience functions are provided in `routing/connect.py`


To make a route, you need to supply:
 - an input port
 - an output port
 - a bend, or a bend factory
 - a straight factory
 - a taper or a taper factory (optional)


To generate a waveguide route:
 1. Generate the backbone of the route. This is a list of manhattan coordinates through which the route would pass through if it used only sharp bends (right angles)

 2. Replace the corners by bend references (with rotation and position computed from the manhattan backbone)

 3. Add tapers if needed and if space permits

 4. generate straight portions in between tapers or bends


The convenience function provided in `routing/connect.py` already have default
parameters and require only an input and an output port to work.


Connecting banks of ports
-------------------------------

Often, several ports have to be linked together without them crossing each other.
One way to tackle simple cases is to use bundle routing.
Several functions are available depending on the use case:

.. autofunction:: pp.routing.route_ports_to_side.route_ports_to_side
.. autofunction:: pp.routing.connect_bundle.link_optical_ports

Example with two arrays of ports connected using `link_optical_ports`

.. plot::
    :include-source:

    import pp
    from pp.component import Port
    from pp.routing.connect_bundle import link_optical_ports

    dy = 200.0
    xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]

    pitch = 10.0
    N = len(xs1)
    xs2 = [-20 + i * pitch for i in range(N // 2)]
    xs2 += [400 + i * pitch for i in range(N // 2)]

    a1 = 90
    a2 = a1 + 180

    ports1 = [Port("top_{}".format(i), (xs1[i], 0), 0.5, a1) for i in range(N)]
    ports2 = [Port("bottom_{}".format(i), (xs2[i], dy), 0.5, a2) for i in range(N)]

    top_cell = pp.Component()
    connections = link_optical_ports(ports1, ports2)
    top_cell.add(connections)
    pp.plotgds(top_cell)


.. autofunction:: pp.routing.connect_bundle.link_ports
.. autofunction:: pp.routing.corner_bundle.corner_bundle
.. autofunction:: pp.routing.u_groove_bundle.u_bundle_direct
.. autofunction:: pp.routing.u_groove_bundle.u_bundle_indirect


 - `route_ports_to_side`, connect all the ports towards one bank of ports facing in one direction
 - `link_optical_ports`, banks or ports facing each other (but with arbitrary and varying pitch on each side)
 - `corner_bundle`, banks of ports with 90Deg / 270Deg between them (again pitch is flexible on both sides)
 - `u_bundle_direct`, banks of ports with direct U-turns
 - `u_bundle_indirect`, banks of ports with indirect U-turns

Each of these cases can either be directly. Another option is to call
`routing/connect_bundle.py:connect_bundle`

Connect bundle acts as a high level entry point. Based on the angle configurations
of the banks of ports, it decides which sub-routine to call between:

 - `link_optical_ports`
 - `corner_bundle`
 - `u_bundle_direct`
 - `u_bundle_indirect`

For now it is not smart enough to decide whether it should call `route_ports_to_side`.
So you either need to connect your ports to face in one direction first, or to
use `route_ports_to_side` prior calling `connect_bundle`

Example of `connect_bundle` behavior when called with two banks of ports
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

.. autofunction:: pp.routing.connect_bundle_from_waypoints.connect_bundle_waypoints

.. plot::
    :include-source:

    import pp
    from pp.routing.connect_bundle_from_waypoints import connect_bundle_waypoints
    @pp.autoname
    def test_connect_bundle_waypoints():
        import pp
        from pp.component import Port

        xs1 = np.arange(10) * 5 - 500.0

        N = xs1.size
        ys2 = np.array([0, 5, 10, 20, 25, 30, 40, 55, 60, 75]) + 500.0

        ports1 = [Port("A_{}".format(i), (xs1[i], 0), 0.5, 90) for i in range(N)]
        ports2 = [Port("B_{}".format(i), (0, ys2[i]), 0.5, 180) for i in range(N)]

        top_cell = pp.Component()
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

        elements = connect_bundle_waypoints(ports1, ports2, way_points)
        top_cell.add(elements)

        return top_cell

    cell = test_connect_bundle_waypoints()
    pp.plotgds(cell)


Connecting optical I/O to a component
--------------------------------------

In cases where individual components have to be tested, a function is provided to
generate the array of optical I/O and connect them to the component. The default connector connects to a 127um pitch fiber array.

.. autofunction:: pp.routing.connect_component.add_io_optical


You can also use individual fibers

.. autofunction:: pp.routing.route_fiber_single
