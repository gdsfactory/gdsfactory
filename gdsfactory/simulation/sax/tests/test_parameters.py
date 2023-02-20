from gdsfactory.simulation.sax.parameter import LithoParameter
import gdsfactory as gf


def test_litho_parameters():
    c = gf.Component("myComponent")
    c.add_polygon(
        [[2.8, 3], [5, 3], [5, 0.8]],
        layer=1,
    )
    c.add_polygon(
        [
            [2, 0],
            [2, 2],
            [4, 2],
            [4, 0],
        ],
        layer=1,
    )
    c.add_polygon(
        [
            [0, 0.5],
            [0, 1.5],
            [3, 1.5],
            [3, 0.5],
        ],
        layer=1,
    )
    c.add_polygon(
        [
            [0, 0],
            [5, 0],
            [5, 3],
            [0, 3],
        ],
        layer=2,
    )
    c.add_polygon(
        [
            [2.5, -2],
            [3.5, -2],
            [3.5, -0.1],
            [2.5, -0.1],
        ],
        layer=1,
    )
    c.add_port(name="o1", center=(0, 1), width=1, orientation=0, layer=1)
    c.add_port(name="o2", center=(3, -2), width=1, orientation=90, layer=1)

    param = LithoParameter(layername="core")
    param.layer_dilation_erosion(c, 0.2)

    param = LithoParameter(layername="core")
    param.layer_dilation_erosion(c, -0.2)

    param = LithoParameter(layername="core")
    param.layer_x_offset(c, 0.5)

    param = LithoParameter(layername="core")
    param.layer_y_offset(c, 0.5)

    param = LithoParameter(layername="core")
    param.layer_round_corners(c, 0.2)
