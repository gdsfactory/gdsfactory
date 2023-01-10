""" Adiabatic coupler as specified by the points in .mat file """

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.types import LayerSpec
from scipy.io import loadmat


@gf.cell
def adiabatic_coupler(
        layer: LayerSpec = "WG",
        port_layer: LayerSpec = "PORT",
        port_width: float = 0.5,
        file: str = 'example_.mat',
) -> Component:
    """Returns an adiabatic coupler with ports: in1, in2, out1, out2.

    Args:
        layer: waveguide layer.
        port_layer: port layer
        port_width: width of the ports
        file: path to .mat file holding the coordinates of the polygon, which implements the adiabatic coupler
    """
    adiabatic_coupler_comp = Component()
    points_top_arm, points_bot_arm = read_points(file=file)
    adiabatic_coupler_comp.add_polygon(points_top_arm, layer=layer)
    adiabatic_coupler_comp.add_polygon(points_bot_arm, layer=layer)
    
    add_ports(adiabatic_coupler_comp, points_top_arm, points_bot_arm, port_layer, port_width)

    return adiabatic_coupler_comp


def add_ports(component, points_top_arm, points_bot_arm, port_layer, port_width):
    """ Adds ports to the adiabatic coupler. Port names are: in1, in2, out1, out2.

    Args:
        points_top_arm: np.array specifying polygon of the top arm of the adiabatic coupler 
        points_bot_arm: np.array specifying polygon of the bottom arm of the adiabatic coupler 
        """
    # find coordinate points of the ports and their indices
    points_port_bot = []
    for ii in range(-1, len(points_bot_arm) - 1):
        current_point = points_bot_arm[ii]
        next_point = points_bot_arm[ii+1]
        if abs(current_point[0] - next_point[0]) < 1e-5:  # check if neighbouring points have the same x-coord
            points_port_bot.append([current_point[0], (current_point[1]+next_point[1])/2])

    points_port_bot = sorted(points_port_bot, key=lambda k: k[0])  # sort the points in the ascending x-coord order
    if len(points_port_bot) != 2: raise ValueError("Coordinates finding of the ports failed")

    points_port_top = []
    for ii in range(-1, len(points_top_arm) - 1):
        current_point = points_top_arm[ii]
        next_point = points_top_arm[ii + 1]
        if abs(current_point[0] - next_point[0]) < 1e-5:  # check if neighbouring points have the same x-coord
            points_port_top.append([current_point[0], (current_point[1] + next_point[1]) / 2])

    points_port_top = sorted(points_port_top, key=lambda k: k[0])  # sort the points in the ascending x-coord order
    if len(points_port_top) != 2: raise ValueError("Coordinates finding of the ports failed")

    # create the ports
    component.add_port(name='in1',  center=points_port_bot[0], orientation=180, layer=port_layer, width=port_width)
    component.add_port(name='out1', center=points_port_bot[1], orientation=0,   layer=port_layer, width=port_width)
    component.add_port(name='in2',  center=points_port_top[0], orientation=180, layer=port_layer, width=port_width)
    component.add_port(name='out2', center=points_port_top[1], orientation=0,   layer=port_layer, width=port_width)


def read_points(file) -> []:
    # load the data and convert points from meters to microns
    ee = loadmat(file)
    cord_bot_wg = ee['bv'] * 1e6
    cord_top_wg = ee['tv'] * 1e6

    duplicate_indices = [
        ii
        for ii, point in enumerate(cord_bot_wg[:-1])
        if abs(point[0] - cord_bot_wg[ii + 1][0]) < 1e-6
        and abs(point[1] - cord_bot_wg[ii + 1][1]) < 1e-6
    ]
    indices_to_be_kept = list(
        set(range(len(cord_bot_wg))) - set(duplicate_indices)
    )
    cord_bot_wg = cord_bot_wg[indices_to_be_kept]

    duplicate_indices = [
        ii
        for ii, point in enumerate(cord_top_wg[:-1])
        if abs(point[0] - cord_top_wg[ii + 1][0]) < 1e-6
        and abs(point[1] - cord_top_wg[ii + 1][1]) < 1e-6
    ]
    indices_to_be_kept = list(
        set(range(len(cord_top_wg))) - set(duplicate_indices)
    )
    cord_top_wg = cord_top_wg[indices_to_be_kept]

    return [cord_bot_wg, cord_top_wg]


if __name__ == "__main__":
    file = 'adiabatic_coupler_files/adiabatic_coupler_example.mat'
    c = adiabatic_coupler(file=file)
    c.show(show_ports=True)
