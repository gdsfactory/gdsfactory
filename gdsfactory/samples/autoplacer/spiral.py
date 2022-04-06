import gdsfactory as gf


@gf.cell
def spiral(N=6, x=50.0) -> gf.Component:
    """Generate a spiral component with grating couplers.

    Args:
        N: number of spiral loops.
        x: inner length.
    """
    c = gf.components.spiral_external_io(N=N, x_inner_length_cutback=x)
    return gf.routing.add_fiber_array(
        component=c, x_grating_offset=-200, fanout_length=30
    )


if __name__ == "__main__":
    c = spiral()
    c.show()
