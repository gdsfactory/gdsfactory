"""fill is now slower.

This code takes now 5 seconds to run.

in version 5.18.6 it was taking 0.66 seconds to run
"""

if __name__ == "__main__":
    import time

    import gdsfactory as gf

    start = time.time()

    coupler_lengths = [10, 20, 30, 40, 50, 60, 70, 80]
    coupler_gaps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    delta_lengths = [10, 100, 200, 300, 400, 500, 500]

    coupler_lengths = [10, 20]
    coupler_gaps = [0.1, 0.2]
    delta_lengths = [10]

    n = 4
    coupler_lengths = [10] * n
    coupler_gaps = [0.2] * n
    delta_lengths = [10] * (n - 1)

    mzi = gf.components.mzi_lattice(
        coupler_lengths=coupler_lengths,
        coupler_gaps=coupler_gaps,
        delta_lengths=delta_lengths,
    )

    # Add fill
    c = gf.Component("component_with_fill")
    layers = [(1, 0)]
    fill_size = [0.5, 0.5]

    c << gf.fill_rectangle(
        mzi,
        fill_size=fill_size,
        fill_layers=layers,
        margin=5,
        fill_densities=[0.8] * len(layers),
        avoid_layers=layers,
    )
    c << mzi
    c.show(show_ports=True)
    end = time.time()
    print(end - start)
