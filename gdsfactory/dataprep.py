try:
    import kfactory as kf
    from kfactory import kdb
    from dask.delayed import delayed, Delayed
    from dask.distributed import Future
    import dask
except ImportError as e:
    print(
        "You can install `pip install gdsfactory[full]` for using dataprep. "
        "And make sure you use python >= 3.10"
    )
    raise e

from gdsfactory.typings import Dict, Layer, PathType


@delayed
def size(region: kdb.Region, offset: float):
    return region.dup().size(int(offset * 1e3))


@delayed
def boolean_or(region1: kdb.Region, region2: kdb.Region):
    return region1.__or__(region2)


@delayed
def boolean_not(region1: kdb.Region, region2: kdb.Region):
    return kdb.Region.__sub__(region1, region2)


@delayed
def copy(region: kdb.Region):
    return region.dup()


class Region(kdb.Region):
    def __iadd__(self, offset):
        """Adds an offset to the layer."""
        return size(self, offset)

    def __isub__(self, offset):
        """Adds an offset to the layer."""
        return size(self, offset)

    def __add__(self, element):
        if isinstance(element, (float, int)):
            return size(self, element)

        elif isinstance(element, (kdb.Region, Delayed)):
            return boolean_or(self, element)
        else:
            raise ValueError(f"Cannot add type {type(element)} to region")

    def __sub__(self, element):
        if isinstance(element, (float, int)):
            return size(self, -element)

        elif isinstance(element, (kdb.Region, Delayed)):
            return boolean_not(self, element)

    def copy(self):
        return self.dup()


class Layout:
    """Layout processor for dataprep.

    Args:
        layermap: dict of layernames to layer numbers {'WG': (1, 0)}
        filepath: to read GDS from.
    """

    def __init__(self, layermap: Dict[str, Layer], filepath: PathType = None):
        lib = kf.kcell.KLib()
        lib.read(filename=str(filepath))
        c = lib[0]

        for layername, layer in layermap.items():
            region = Region()
            layer = lib.layer(layer[0], layer[1])
            region.insert(c.begin_shapes_rec(layer))
            region.merge()
            setattr(self, layername, region)

        self.layermap = layermap
        self.lib = lib

    def calculate(self):
        tasks = {layername: getattr(self, layername) for layername in self.layermap}
        results = dask.compute(tasks)
        for layername, result in results[0].items():
            setattr(self, layername, result)

    def visualize(self, filename):
        """Visualize task graph."""
        tasks = []
        layer_names = []
        named_tasks = {}
        print("visualizing task graph...")
        for layername in self.layermap.keys():
            region = getattr(self, layername)
            if isinstance(region, (Delayed, Future)):
                tasks.append(region)
                layer_names.append(layername)
                named_tasks[layername] = region
        dask.visualize(named_tasks, filename=filename)

    def write(self, filename, cellname: str = "out") -> kf.KCell:
        """Write gds.

        Args:
            filepath: gdspath.
            cellname: for top cell.
        """
        self.calculate()
        c = kf.KCell(cellname, self.lib)

        for layername, layer in self.layermap.items():
            region = getattr(self, layername)
            if isinstance(region, Delayed):
                region = region.compute()
            try:
                c.shapes(self.lib.layer(layer[0], layer[1])).insert(region)
            except TypeError:
                raise ValueError(
                    f"Unexpected type for region {layername}: {type(region)}"
                )
        c.write(filename)
        return c

    def __delattr__(self, element):
        setattr(self, element, Region())
