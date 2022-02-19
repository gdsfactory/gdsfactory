if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    import gdsfactory.simulation.gtidy3d as gt

    nm = 1e-3
    wavelength = np.linspace(1500, 1600) * nm
    plt.plot(wavelength, gt.materials.get_index("si"))
    plt.title("cSi crystalline silicon")
    plt.xlabel("wavelength")
    plt.ylabel("n")
