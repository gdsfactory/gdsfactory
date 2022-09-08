port_symmetries_1x1 = {
    "o1@0,o1@0": ["o2@0,o2@0"],
    "o2@0,o1@0": ["o1@0,o2@0"],
}


port_symmetries_crossing = {
    "o1@0,o1@0": ["o2@0,o2@0", "o3@0,o3@0", "o4@0,o4@0"],
    "o2@0,o1@0": ["o1@0,o2@0", "o3@0,o4@0", "o4@0,o3@0"],
    "o3@0,o1@0": ["o1@0,o3@0", "o2@0,o4@0", "o4@0,o2@0"],
    "o4@0,o1@0": ["o1@0,o4@0", "o2@0,o3@0", "o3@0,o2@0"],
}


if __name__ == "__main__":
    import numpy as np

    port_symmetries = port_symmetries_crossing

    sp = dict(wavelengths=np.linspace(1.5, 1.6, 3))
    sp["o1@0,o1@0"] = 2 * np.linspace(1.5, 1.6, 3)

    for key, symmetries in port_symmetries.items():
        for sym in symmetries:
            if key in sp:
                sp[sym] = sp[key]
