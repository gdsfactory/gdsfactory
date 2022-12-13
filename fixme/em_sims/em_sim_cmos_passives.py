# Simulate CMOS passives in open source em tools (ASITIC, meep, pyems etc)
# 1. Provide an interface with basic em simulation with minimal settings such as meshing and accuracy and provide
# reasonable defaults
# 2. Write output s-parameters or simulatable broadband spice model that can be plugged in ngspice/Xyce
# 3. Compare with ASITIC created s-parameters for fine tuning ()
# 4. Current example set on skywater130, once proven extend to gf180 and sky90(when available)


import gdsfactory as gf


def import_inductor_gds(input_file, topcell) -> gf.Component:
    """Import inductor gds to the database and returns a Component.

    Args:
        input_file:
        topcell:

    Returns: Component

    """
    inductor_in = gf.import_gds(input_file, cellname=topcell, flatten=True)
    print("Inductor object : ", inductor_in)
    # lc = ind_comp << inductor_in
    return gf.Component("Inductor Top")


def extract_shapes() -> dict:
    """Extract the shapes from the gds.

    Args:
        ind_comp: Provide inductor components.

    Returns: Polygons dict in the db format

    """
    # rect_dict = {}
    lc_inst = import_inductor_gds("inductor_cell.gds", "diff_octagon_inductor")
    # for lay, shps in lc_inst.get_polygons(True).items():
    #    for poly in shps:
    #        print(poly)
    #        print(shpoly(poly).exterior)
    #        exit()
    #        if lay not in rect_dict.keys():
    #            rect_dict[lay] = [minimum_bounding_rectangle(poly)]
    #        else:
    #            rect_dict[lay].append(minimum_bounding_rectangle(poly))
    return lc_inst.get_polygons(True)
    # return rect_dict


def test_polygon():
    poly_dict = extract_shapes()
    # plot_polygon(poly_dict)
    npoly_dict = {}
    for k, v in poly_dict.items():
        for i in v:
            if len(i) != 4:
                if k in npoly_dict:
                    npoly_dict[k].append(i)
                else:
                    npoly_dict[k] = [i]
                    # assert len(i) == 4, plot_polygon(k, i)
                    # f'{len(i)}\t{i}\tThese are not rectangles'; plot_polygon(k, i)
                    # assert len(i) == 4,  ;(i, k)
    plot_shapes(npoly_dict)


def plot_shapes(poly_dict):
    test_comp = gf.Component("Test debug Comp")
    for layer, ply in poly_dict.items():
        test_comp.add_polygon(points=ply, layer=layer)
    test_comp.plot(alpha=0.1)


def plot_polygon(lay, poly):
    test_comp = gf.Component("Test debug Comp")
    test_comp.add_polygon(points=poly, layer=lay)
    test_comp.plot()


# Test the polygons being imported and plot them
test_polygon()
