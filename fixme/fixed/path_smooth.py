import gdsfactory as gf

p = gf.path.smooth(points=[(0, 0), (0, 1000), (1, 10000)])

# IndexError: index -1 is out of bounds for axis 0 with size 0
c = p.extrude(cross_section="strip")
c.show()
