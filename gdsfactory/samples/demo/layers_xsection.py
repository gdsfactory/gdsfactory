import gdsfactory as gf

if __name__ == "__main__":
    c = gf.components.straight_heater_doped_rib()
    c.show()

    scene = c.to_3d()
    scene.show()
