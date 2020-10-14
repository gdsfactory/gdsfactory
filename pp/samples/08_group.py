import pp

D = pp.Component()

t1 = D << pp.c.text("1")
t2 = D << pp.c.text("2")
t3 = D << pp.c.text("3")
t4 = D << pp.c.text("4")
t5 = D << pp.c.text("5")
t6 = D << pp.c.text("6")

D.distribute(direction="x", spacing=3)

pp.show(D)
