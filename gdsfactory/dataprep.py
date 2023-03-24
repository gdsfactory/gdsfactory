try:
    import kfactory as kf
    from kfactory import kdb

    class Region(kdb.Region):
        def __iadd__(self, offset):
            """Adds an offset to the layer."""
            return self.size(int(offset * 1e3))

        def __isub__(self, offset):
            """Adds an offset to the layer."""
            return self.size(-int(offset * 1e3))

        def __add__(self, element):
            if isinstance(element, float):
                self.size(int(element * 1e3))

            elif isinstance(element, kdb.Region):
                self = self.__or__(element)

        def __sub__(self, element):
            if isinstance(element, float):
                self.size(-int(element * 1e3))

            elif isinstance(element, kdb.Region):
                return super().__sub__(element)

        def copy(self):
            return self.dup()

except ImportError:
    print(
        "You can install `pip install gdsfactory[full]` for using dataprep."
        "And make sure you use python >= 3.10"
    )


class Layout:
    def __init__(self, layermap, filepath=None):
        lib = kf.kcell.KLib()
        lib.read(filename=filepath)
        c = lib[0]

        for layername, layer in layermap.items():
            region = Region()
            layer = lib.layer(layer[0], layer[1])
            region.insert(c.begin_shapes_rec(layer))
            region.merge()
            setattr(self, layername, region)

        self.layermap = layermap
        self.lib = lib

    def write(self, filename, cellname: str = "out") -> kf.KCell:
        c = kf.KCell(cellname, self.lib)

        for layername, layer in self.layermap.items():
            region = getattr(self, layername)
            c.shapes(self.lib.layer(layer[0], layer[1])).insert(region)
        c.write(filename)
        return c

    def __delattr__(self, element):
        setattr(self, element, Region())
