from gdsfactory.generic_tech.layer_map import LAYER as l
import gdsfactory.dataprep as dp
import gdsfactory as gf
import dask

dask.config.set(scheduler='threads')

if __name__ == '__main__':
    c = gf.c.coupler_ring(cross_section="strip")
    c.write_gds("src.gds")

    d = dp.Layout(filepath="src.gds", layermap=dict(l))
    # we're going to do a bunch of derivations just to get a more interesting task graph... don't mind if it physically doesn't make sense
    d.SLAB150 = d.WG + 3
    d.SHALLOW_ETCH = d.SLAB150 - d.WG
    d.DEEP_ETCH = d.WG + 2
    d.M1 = d.DEEP_ETCH + 1
    d.M2 = d.DEEP_ETCH - d.SHALLOW_ETCH
    # visualize the taskgraph and save as 'tasks.html'
    d.visualize("tasks")
    # evaluation of the task graph is lazy
    d.calculate()
    c = d.write("dst.gds")
    # c
