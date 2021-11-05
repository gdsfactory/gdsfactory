Routing
===================================================================================================

get_route
---------------------------------------------------------------------------------------------------

.. automodule:: gdsfactory.routing.get_route
   :members:


get_route_from_steps
---------------------------------------------------------------------------------------------------

.. automodule:: gdsfactory.routing.get_route_from_steps
   :members:


get_bundle
---------------------------------------------------------------------------------------------------

Often, several ports have to be linked together without them crossing each other. One way to tackle
simple cases is to use bundle routing. Several functions are available depending on the use case:

.. autofunction:: gdsfactory.routing.get_bundle.get_bundle

Example with two arrays of ports connected using `get_bundle`

.. plot::
    :include-source:

    import gdsfactory as gf

    @gf.cell
    def test_north_to_south():
        dy = 200.0
        xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]

        pitch = 10.0
        N = len(xs1)
        xs2 = [-20 + i * pitch for i in range(N // 2)]
        xs2 += [400 + i * pitch for i in range(N // 2)]

        a1 = 90
        a2 = a1 + 180

        ports1 = [gf.Port("top_{}".format(i), (xs1[i], 0), 0.5, a1) for i in range(N)]
        ports2 = [gf.Port("bottom_{}".format(i), (xs2[i], dy), 0.5, a2) for i in range(N)]

        c = gf.Component()
        routes = gf.routing.get_bundle(ports1, ports2)
        for route in routes:
            c.add(route.references)

        return c


    c = test_north_to_south()
    c.show()
    c.plot()


`get bundle` is the generic river routing function that will call different function depending on
the port orientation. Get bundle acts as a high level entry point. Based on the angle
configurations of the banks of ports, it decides which sub-routine to call:

 - `get_bundle_same_axis`, banks or ports facing each other (but with arbitrary and varying pitch
   on each side)
 - `get_bundle_corner`, banks of ports with 90Deg / 270Deg between them (again pitch is flexible on
   both sides)
 - `get_bundle_udirect`, banks of ports with direct U-turns
 - `get_bundle_uindirect`, banks of ports with indirect U-turns

 Or you can also call each functions individually

.. autofunction:: gdsfactory.routing.get_bundle.get_bundle_same_axis
.. autofunction:: gdsfactory.routing.get_bundle_corner.get_bundle_corner
.. autofunction:: gdsfactory.routing.get_bundle_u.get_bundle_udirect
.. autofunction:: gdsfactory.routing.get_bundle_u.get_bundle_uindirect

get_bundle_from_steps
---------------------------------------------------------------------------------------------------

.. autofunction:: gdsfactory.routing.get_bundle_from_steps.get_bundle_from_steps


route_ports_to_side
---------------------------------------------------------------------------------------------------

For now `get_bundle` is not smart enough to decide whether it should call `route_ports_to_side`.
So you either need to connect your ports to face in one direction first, or to
use `route_ports_to_side` before calling `get_bundle`

.. autofunction:: gdsfactory.routing.route_ports_to_side.route_ports_to_side



get_bundle_from_waypoints
---------------------------------------------------------------------------------------------------

.. autofunction:: gdsfactory.routing.get_bundle_from_waypoints.get_bundle_from_waypoints

.. plot::
    :include-source:

    import numpy as np
    import gdsfactory as gf


    @gf.cell
    def test_connect_bundle_waypoints():
        """Connect bundle of ports with bundle of routes following a list of waypoints."""
        xs1 = np.arange(10) * 5 - 500.0
        N = xs1.size
        ys2 = np.array([0, 5, 10, 20, 25, 30, 40, 55, 60, 75]) + 500.0

        ports1 = [gf.Port(f"A_{i}", (xs1[i], 0), 0.5, 90) for i in range(N)]
        ports2 = [gf.Port(f"B_{i}", (0, ys2[i]), 0.5, 180) for i in range(N)]

        c = gf.Component()
        waypoints = [
            ports1[0].position + (0, 100),
            ports1[0].position + (200, 100),
            ports1[0].position + (200, -200),
            ports1[0].position + (0, -200),
            ports1[0].position + (0, -350),
            ports1[0].position + (400, -350),
            (ports1[0].x + 400, ports2[0].y),
        ]

        routes = gf.routing.get_bundle_from_waypoints(ports1, ports2, waypoints)
        for route in routes:
            c.add(route.references)

        return c

    cell = test_connect_bundle_waypoints()
    cell.plot()



get_bundle_path_length_match
---------------------------------------------------------------------------------------------------


.. autofunction:: gdsfactory.routing.get_bundle_path_length_match.get_bundle_path_length_match


add_fiber_array / add_fiber_single
---------------------------------------------------------------------------------------------------

In cases where individual components have to be tested, you can generate the array of optical I/O
and connect them to the component.

You can connect the waveguides to a 127um pitch fiber array.

.. autofunction:: gdsfactory.routing.add_fiber_array.add_fiber_array


Or can also connect to individual fibers for input and output.

.. autofunction:: gdsfactory.routing.add_fiber_single.add_fiber_single
