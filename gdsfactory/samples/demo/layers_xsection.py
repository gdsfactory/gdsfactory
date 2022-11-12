import gdsfactory as gf

if __name__ == "__main__":
    c = gf.components.mzi_phase_shifter_top_heater_metal()
    # c = gf.routing.add_fiber_array(c)
    # c = gf.routing.add_electrical_pads_top(c)
    c.show()

    # scene = c.to_3d()
    # scene.show()
