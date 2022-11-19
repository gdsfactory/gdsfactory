import gdsfactory as gf

comp1 = gf.components.taper(length=100, width1=20, width2=50)
outline_comp1 = gf.geometry.outline(comp1, open_ports=True, precision=1e-3)
outline_comp1.show()
