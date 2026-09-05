<!-- #region -->
## 7 to 8


Gdsfactory v8, based on KLayout instead of gdstk.
KLayout offers enhanced routing functions and additional features such DRC, dummy fill, and connectivity checks.

For those still using gdsfactory v7, it is hosted on [https://github.com/gdsfactory/gdsfactory7](https://github.com/gdsfactory/gdsfactory7), along with the [documentation](https://gdsfactory.github.io/gdsfactory7/).

### Benefits of Migrating:

- **Integrated Data**: Ports, information, and settings are stored within the GDS file, eliminating the need for separate files.
- **Improved Booleans**: Booleans are more robust with integer-based polygons, removing slivers of less than 1nm.
- **Enhanced Features**: More robust booleans, DRC, LVS, and connectivity checks.
- **Active Maintenance**: KLayout is more actively maintained with frequent updates than gdstk.
- **Advanced Routing Algorithms**: Better routing algorithms for efficient design. Thanks to Kfactory.
- **Grid Alignment**: Ports and polygons snap to grid by default, reducing the likelihood of 1nm gaps.

### Drawbacks of Migrating:

- **Potential Errors**: As with any code changes, undesired errors may be introduced. It is recommended to have regression tests for all your components before you migrate.
- **Non-Manhattan Placement**: Slightly more difficult to define non-manhattan references, routes, or port. Manhattan Components (at 0, 90, 180 and 270) work the same way.
- **Incomplete Functionality**: Some features, such as `route_path_length_match`, are not yet implemented.

### Major Differences:

- **Port units**: `port.x`, `port.x`, `port.width` have changed to Database Units (1nm by default). To set/get them in um, use `d` (decimal) e.g., `port.dxmin` `port.dcenter` or `port.dwidth` which are floats.
- **LayerMap**: Now an Enum of integers.
- **Routing Functions**: New functions do not require starting ports to be in the same orientation and monitor for self-intersections. `get_route` is now `route_bundle`, and `get_bundle` is now `route_bundle`.
- **Grid Snapping**: All polygon points snap to grid, mitigating 1nm gaps.

### Minor Differences:

- Replace `from gdsfactory.cell import cell` with `from gdsfactory import cell`.
<!-- #endregion -->

<!-- #region -->
## LayerMap 

In v7 or below, a LayerMap needs to be called.

```python

from gdsfactory.technology import LayerMap

class LayerMapFab(LayerMap):
    WG = (1, 0)

LAYER = LayerMapFab()
```

However in v8 it has a different type and does not need to be called.

```python

from gdsfactory.technology import LayerMap

class LayerMapFab(LayerMap):
    WG = (1, 0)

LAYER = LayerMapFab
```

See below:
<!-- #endregion -->

```python
import gdsfactory as gf
from gdsfactory.technology import LayerMap




gf.gpdk.PDK.activate()
class LayerMapFab(LayerMap):
    WG = (1, 0)


LAYER = LayerMapFab
type(LAYER)

```

```python
LAYER.WG
```

```python
tuple(LAYER.WG)
```

```python
str(LAYER.WG)
```

## Ports

Ports are now stored in the GDS file and are not stored in a separate file.

Ports are snapped to grid, and therefore, width, x and y is in DBU (1nm by default).

```python
c = gf.components.straight(length=10)
print(c.ports["o2"].width, "in DBU")
print(c.ports["o2"].dwidth, "in um")
```

```python
print(c.ports["o2"].dx, "in DBU")
print(c.ports["o2"].x, "in um")
```

## Component.bbox is now a function

Both Components and Instances have a `bbox` and `dbbox` that are now functions.

```python
c = gf.c.straight(length=10)
c.bbox()
```

```python
dir(c.dbbox())
```

```python
c.dbbox().right
```

For the old bbox behavior you can use `c.bbox_np` which returns the bbox as a numpy array.

```python
c.bbox_np()
```

## Component.get_polygons()

`Component.get_polygons()` returns all the Polygons and can be slow.

`Component.get_polygons_points()` is the equivalent to `Component.get_polygons()` in gdsfactory7 where we return the polygon edges.

```python
import gdsfactory as gf

c = gf.pack([gf.components.seal_ring_segmented()] * 200)[0]
c
```

```python
%time
p = c.get_polygons_points()
```

If you only care about the polygons from one layer you can also only extract those.

```python
%time
p = c.get_polygons_points(layers=("M3",))
```

## ref.get_ports_list

Reference does not have `get_ports_list`. You can use

- ref.ports to get all ports
- ref.ports.filter(...) to filter all ports with angle, port_type, orientation ...
- gf.port.get_ports_list(b, ...) for backwards compatibility.

```python
import gdsfactory as gf

c = gf.Component()
bend = gf.components.bend_euler()
b = c.add_ref(bend)

b.ports.filter(orientation=90)
```

```python
gf.port.get_ports_list(b, orientation=90)
```

## Routing functions

Routing functions NO longer return the route Instances but they place the instances in a Component, so you have to pass a Component as the first argument.

```python
c = gf.Component()
w = gf.components.straight(cross_section="rib")
top = c << w
bot = c << w
bot.move((0, -2))

p0 = top.ports["o2"]
p1 = bot.ports["o2"]

r = gf.routing.route_bundle(
    c,
    p0,
    p1,
    cross_section="rib",
)
c
```

🚀 The new routing functions allow the starting ports `ports1` to have different orientations.

The end ports `ports2` still require to have the same orientation.

```python
c = gf.Component()
top = c << gf.components.nxn(north=8, south=0, east=0, west=0)
bot = c << gf.components.nxn(north=2, south=2, east=2, west=2, xsize=10, ysize=10)
top.movey(100)

routes = gf.routing.route_bundle(
    c,
    ports1=bot.ports,
    ports2=top.ports,
    radius=5,
    sort_ports=True,
    cross_section="strip",
)
c
```

## Built in connectivity checks

gdsfactory8 includes connectivity checks to ensure things are properly connected. no

This can help you find any:

- Gaps between ports.
- Width mismatches.
- Unconnected ports.

```python
c = gf.Component()
s = c << gf.components.straight(width=1)

b1 = c << gf.components.bend_euler()
b1.connect("o1", s["o2"], allow_width_mismatch=True)

b2 = c << gf.components.bend_euler(radius=5)
b2.connect("o1", s["o1"], allow_width_mismatch=True)

# Grating Coupler: This is a device on a photonic chip that acts as an interface, 
# scattering light to allow it to travel between an optical fiber (off-chip) and a waveguide (on-chip).
# TE: This stands for Transverse-Electric, which refers to a specific polarization of light. 
# This coupler is optimized to work efficiently with TE-polarized light.
gc = gf.components.grating_coupler_elliptical_te()
gc1 = c << gc
gc2 = c << gc

# Two grating couplers (gc1, gc2) are added. They are then connected to the open ends of the bends, completing the U-shaped path for the light.
gc1.connect("o1", b1.ports["o2"])
gc2.connect("o1", b2.ports["o2"])

c.plot()
```

```python
# The connectivity_check function is run on the component c. 
# It searches for all ports of the specified types ("optical", "electrical") that are not connected to another port. 
# The locations of these unspecified ports are stored in the lyrdb object.
lyrdb = c.connectivity_check(port_types=("optical", "electrical"))

# The results of the check are saved to a file named errors.lyrdb. 
# This is a KLayout Layer Properties file that contains instructions on how to draw markers at the locations of the unconnected ports.
filepath = gf.config.home / "errors.lyrdb"
lyrdb.save(filepath)

# The gf.show function opens the component c in the KLayout viewer. The lyrdb=filepath argument tells KLayout to also load the error report file. 
# This causes KLayout to place visual markers (often red squares or circles) directly on the layout at the exact location of each unconnected port,
# making them easy to find and fix.
gf.show(c, lyrdb=filepath)
```

<!-- #region -->
## Metadata

GDSFactory adds metadata to the gds files to store the settings and the ports. If you are using a different EDA tool, you disable the metadata by setting `with_metadata=False` when you run `write_gds`.

```python
import gdsfactory as gf

c = gf.components.mzi()
c.write_gds("mzi.gds", with_metadata=False)
```
<!-- #endregion -->
