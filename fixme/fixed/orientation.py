import gdsfactory as gf

c = gf.Component("pads_no_orientation")
pt = c << gf.components.pad()
pb = c << gf.components.pad()
pb.connect("pad", pt["pad"])
c.show()
