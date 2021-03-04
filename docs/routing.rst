Routing
=============================

Route two ports with manhattan route
----------------------------------------------

.. automodule:: pp.routing.get_route


Route two lists of ports with a list of routes (bundle)
---------------------------------------------------------------

Often, several ports have to be linked together without them crossing each other.
One way to tackle simple cases is to use bundle routing.
Several functions are available depending on the use case:

.. autofunction:: pp.routing.get_bundle.get_bundle

Example with two arrays of ports connected using `get_bundle`

.. plot::
    :include-source:

    import pp

    @pp.cell
    def test_north_to_south():
        dy = 200.0
        xs1 = [-500, -300, -100, -90, -80, -55, -35, 200, 210, 240, 500, 650]

        pitch = 10.0
        N = len(xs1)
        xs2 = [-20 + i * pitch for i in range(N // 2)]
        xs2 += [400 + i * pitch for i in range(N // 2)]

        a1 = 90
        a2 = a1 + 180

        ports1 = [pp.Port("top_{}".format(i), (xs1[i], 0), 0.5, a1) for i in range(N)]
        ports2 = [pp.Port("bottom_{}".format(i), (xs2[i], dy), 0.5, a2) for i in range(N)]

        c = pp.Component()
        routes = pp.routing.get_bundle(ports1, ports2)
        for route in routes:
            c.add(route['references'])

        return c


    c = test_north_to_south()
    c.show()
    c.plot()



Route to fiber I/O
--------------------------------------

In cases where individual components have to be tested, a function is provided to
generate the array of optical I/O and connect them to the component. The default connector connects to a 127um pitch fiber array.

.. autofunction:: pp.routing.add_fiber_array.add_fiber_array


You can also use individual fibers

.. autofunction:: pp.routing.add_fiber_single.add_fiber_single
