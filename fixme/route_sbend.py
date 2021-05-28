"""
Route manhattan sometimes does not fit a route.
it would be nice to enable Sbend routing for those cases

"""
import pp
from pp.routing.manhattan import route_manhattan


if __name__ == "__main__":
    c = pp.Component()

    inputs = [
        pp.Port("in6", midpoint=(0, 0), width=0.5, orientation=0),
    ]

    outputs = [
        pp.Port("in6", midpoint=(10, 5), width=0.5, orientation=180),
    ]

    lengths = [1] * len(inputs)

    for input_port, output_port, length in zip(inputs, outputs, lengths):

        route = route_manhattan(
            input_port=input_port,
            output_port=output_port,
            waveguide="nitride",
            radius=5.0,
        )

        c.add(route.references)

    c.show()
