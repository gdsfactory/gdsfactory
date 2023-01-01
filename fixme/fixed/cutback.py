import gdsfactory as gf

cross_section = "rib"

b = gf.components.bezier(
    control_points=[[0.0, 0.0], [6.335, 0.0], [10, 4.665], [10, 10.0]],
    npoints=201,
    with_manhattan_facing_angles=True,
    with_bbox=True,
    cross_section=cross_section,
)  # b is a bend Component

b_function = gf.partial(
    gf.components.bezier,
    control_points=[[0.0, 0.0], [6.335, 0.0], [10, 4.665], [10, 10.0]],
    npoints=201,
    with_manhattan_facing_angles=True,
    with_bbox=True,
    cross_section=cross_section,
)  # b_function is a PCell that returns bend components

c = gf.components.cutback_bend90(
    bend90=b_function,
    straight_length=10,
    rows=8,
    columns=3,
    spacing=20,
    cross_section=cross_section,
)
c.show()
