# Routing API

## route_single

::: gdsfactory.routing.route_single

::: gdsfactory.routing.route_quad

::: gdsfactory.routing.route_sharp

## route_bundle

When you need to route groups of ports together without them crossing each other you can use a bundle/river/bus router.
`route_bundle` is the generic river bundle bus routing function that will call different functions depending on
the port orientation. Get bundle acts as a high level entry point. Based on the angle
configurations of the banks of ports, it decides which sub-routine to call:

::: gdsfactory.routing.route_bundle

::: gdsfactory.routing.route_bundle_electrical

## route_bundle_all_angle

::: gdsfactory.routing.route_bundle_all_angle

## route_ports_to_side

For now `route_bundle` is not smart enough to decide whether it should call `route_ports_to_side`.
So you either need to connect your ports to face in one direction first, or to
use `route_ports_to_side` before calling `route_bundle`.

::: gdsfactory.routing.route_ports_to_side

::: gdsfactory.routing.route_ports_to_x

::: gdsfactory.routing.route_ports_to_y

::: gdsfactory.routing.route_south

## fanout

::: gdsfactory.routing.fanout2x2

## add_fiber_array

In cases where individual components have to be tested, you can generate the array of optical I/O and connect them to the component.

You can connect the waveguides to a 127um pitch fiber array or to individual fibers for input and output.

::: gdsfactory.routing.add_fiber_array.add_fiber_array

## add_pads

::: gdsfactory.routing.add_pads_top

::: gdsfactory.routing.add_pads_bot

::: gdsfactory.routing.add_electrical_pads_shortest

::: gdsfactory.routing.add_electrical_pads_top

::: gdsfactory.routing.add_electrical_pads_top_dc
