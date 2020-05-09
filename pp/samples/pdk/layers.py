from collections import namedtuple

layermap = dict(WG=(1, 0), DEVREC=(68, 0), LABEL=(10, 0), PORT=(1, 10))

LAYER = namedtuple("layer", layermap.keys())(*layermap.values())


if __name__ == "__main__":
    print(LAYER)
