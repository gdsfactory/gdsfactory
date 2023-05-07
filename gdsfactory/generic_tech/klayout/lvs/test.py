import gdsfactory as gf

c = gf.Component()
c1 = c << gf.components.straight_heater_metal(length=50, heater_width=10)
c2 = c << gf.components.straight_heater_metal(length=100, heater_width=10)
c3 = c << gf.components.straight_heater_metal(length=100, heater_width=5)
c4 = c << gf.components.straight_heater_metal(length=50, heater_width=10)
c1.xmin = 0
c2.xmin = c1.xmin
c3.xmin = c1.xmin
c4.xmin = c1.xmin

c1.ymin = 0
c2.ymin = c1.ymax + 50
c3.ymin = c2.ymax + 50
c4.ymin = c3.ymax + 50

c.show()
