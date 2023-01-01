import gdsfactory as gf

d = gf.components.bezier(
    control_points=[[0.0, 0.0], [6.335, 0.0], [10, 4.665], [10, 10.0]],
    npoints=201,
    with_manhattan_facing_angles=True,
    with_bbox=True,
    cross_section="strip",
)
d.show()

b = gf.components.bend_circular(cross_section="strip")
b.show()

s = gf.components.straight(cross_section="strip")
s.show()

c = gf.components.cutback_bend90(
    bend90=b, straight=s, straight_length=10, rows=8, columns=3, spacing=20
)
c.show()


# c = gf.components.cutback_bend90(
#     bend90=b, straight=b, straight_length=10, rows=8, columns=3, spacing=20
# )
# c.show()

c = gf.components.cutback_bend90(bend90=b, cross_section="strip")
c.show()
