# import matplotlib.pyplot as plt
import meep as mp


def plot_xsection(sim, center=(0, 0, 0), size=(0, 2, 2)):
    """
    sim: simulation object
    """
    sim.plot2D(output_plane=mp.Volume(center=center, size=size))
    # plt.colorbar()
