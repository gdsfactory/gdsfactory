# Geometry operations

gdsfactory provides you with some geometric functions


## Boolean
There are several common boolean-type operations available in the geometry library.  These include typical boolean operations (and/or/not/xor), offsetting (expanding/shrinking polygons), outlining, and inverting.



The ``gf.boolean()`` function can perform AND/OR/NOT/XOR operations, and will return a new geometry with the result of that operation.

```python
import gdsfactory as gf


gf.gpdk.PDK.activate()
E = gf.components.ellipse(radii=(10, 5), layer=(1, 0))
R = gf.components.rectangle(size=(15, 5), layer=(2, 0))

# A new component C is created by performing a boolean operation.
# A=E, B=R: E is the shape to be subtracted from, and R is the shape to subtract.
# operation="not": This specifies the subtraction operation (A - B).
# layer1=(1, 0), layer2=(2, 0): The operation uses the shapes from layer (1, 0) of component E and layer (2, 0) of component R.
# layer=(3, 0): The final, resulting shape is placed on a new layer, (3, 0).
C = gf.boolean(A=E, B=R, operation="not", layer1=(1, 0), layer2=(2, 0), layer=(3, 0))


# Other operations include 'and', 'or', 'xor', or equivalently 'A-B', 'B-A', 'A+B'.

# Plot the originals and the result.
D = gf.Component()
D.add_ref(E)
D.add_ref(R)
D.add_ref(C).movex(30)
D.plot()

```

To learn how booleans work you can try all the different operations `not`, `and`, `or`, `xor`.

```python
operation = "not" # This is subtraction (A - B). It keeps the parts of shape A that do not overlap with shape B.
operation = "and" # This is intersection (A ∩ B). It keeps only the parts where shape A and shape B overlap.
operation = "or"  # This is union (A ∪ B). It combines shape A and shape B into a single, merged shape.
operation = "xor" # This is exclusive OR (symmetric difference). It keeps the parts of shape A and shape B that do not overlap.

r1 = (8, 8)
r2 = (11, 4)
r1 = (80, 80)
r2 = (110, 40)

angle_resolution = 0.1

c1 = gf.components.ellipse(radii=r1, layer=(1, 0), angle_resolution=angle_resolution)
c2 = gf.components.ellipse(radii=r2, layer=(1, 0), angle_resolution=angle_resolution)
```

```python
c4 = gf.boolean(c1, c2, operation=operation, layer=(1, 0))
c4.plot()
```

## Flatten

Hierarchical GDS uses references to optimize memory usage, but there are times when you may want to merge all polygons. In such cases, you can flatten the GDS to absorb all the references.

```python
import gdsfactory as gf

c = gf.Component()
e = c << gf.components.ellipse(
    radii=(10, 5), layer=(1, 0)
)  # Ellipse. equivalent to c.add_ref(gf.components.ellipse(radii=(10, 5), layer=(1, 0)).
r = c << gf.components.rectangle(
    size=(15, 5), layer=(2, 0)
)  # Rectangle. equivalent to c.add_ref(gf.components.rectangle(size=(15, 5), layer=(2, 0)).
print(len(c.insts))  # 2 component instances.

# This command flattens the component c, removing all levels of hierarchy.
# It replaces all instances (references) of sub-components with copies of their raw geometric shapes.
c.flatten()
print(len(c.insts))  # 0 in the flattened component.
c
```

## Fix min space violations

You can use `fix_spacing` to fix min space violations.

```python
c = gf.components.coupler_ring(cross_section='rib')
c
```

```python
c = c.copy()
c.fix_spacing(layer=(3,0), min_space=1)
c
```

## Fix min width violations

Some PDKs have minimum width rules, where a geometry is too narrow and needs to be widened. GDSFactory provides a function to fix this.

```python
c = gf.c.taper(width1=10, width2=1, length=5)
c
```

```python
c = gf.c.taper(width1=10, width2=1, length=5).copy()
c.fix_width(layer=(1, 0), min_width=1)
c
```

## Offset

```python
import gdsfactory as gf
from gdsfactory.gpdk import LAYER

c = gf.components.coupler_ring()
c
```

```python
r = c.get_region(LAYER.WG)
r
```

```python
c2 = c.copy()
r = r.sized(2000)  # Note that the region is sized in DB units (1nm).
c2.add_polygon(r, layer=LAYER.SLAB90)
c2
```

To avoid acute angles you can run over / under (dilation+erosion).

```python
c2 = c.copy()

d = 800  # DB units
r = r.sized(+d + 2000)
r = r.sized(-d)
c2.add_polygon(r, layer=LAYER.SLAB90)
c2
```

You can also apply it to a component directly.

```python
c = gf.Component()
core = gf.components.coupler_ring()

clad = core.copy()
clad.offset(layer="WG", distance=1)
clad.remap_layers({"WG": "SLAB90"})

c.add_ref(core)
c.add_ref(clad)

c
```

```python
c.over_under(layer="SLAB90", distance=0.2)
c
```

## Outline

Notice that all operations with regions are in database units, where 1DBU is usually 1nm.

```python
c2 = gf.Component()

d = 800
r = c.get_region(layer=LAYER.WG)
r_sized = r.sized(+2000)

r_outline = r_sized - r
c2.add_polygon(r_outline, layer=LAYER.SLAB90)
c2
```

## Round corners

```python
c = gf.components.triangle()
c
```

```python
c2 = gf.Component()

rinner = 1000  # 	The circle radius of inner corners (in database units).
router = 1000  # 	The circle radius of outer corners (in database units).
n = 300  # 	The number of points per full circle.

# Round corners for one layer only.
for p in c.get_polygons()[LAYER.WG]:
    p_round = p.to_simple_polygon().round_corners(rinner, router, n)
    c2.add_polygon(list(p_round.each_point()), layer=LAYER.WG)

c2
```

```python
c = gf.Component()
t = c << gf.components.triangle(x=10, y=20, layer="WG")
t = c << gf.components.triangle(x=20, y=40, layer="SLAB90")

c2 = gf.Component()
rinner = 1000 
router = 1000 
n = 300 

# Round corners for all layers.
for layer, polygons in c.get_polygons().items():
    for p in polygons:
        p_round = p.to_simple_polygon().round_corners(rinner, router, n)
        c2.add_polygon(list(p_round.each_point()), layer=layer)

c2
```

## Union

```python
import gdsfactory as gf

# This code creates a component containing six overlapping ellipses,
# each on a different layer and each rotated by an incrementally larger angle, creating a flower-like pattern.
c = gf.Component()
e0 = c << gf.components.ellipse(layer=(1, 0))
e1 = c << gf.components.ellipse(layer=(2, 0))
e2 = c << gf.components.ellipse(layer=(3, 0))
e3 = c << gf.components.ellipse(layer=(4, 0))
e4 = c << gf.components.ellipse(layer=(5, 0))
e5 = c << gf.components.ellipse(layer=(6, 0))

e1.rotate(15 * 1)
e2.rotate(15 * 2)
e3.rotate(15 * 3)
e4.rotate(15 * 4)
e5.rotate(15 * 5)

c
```

```python
r = gf.Region()

for layer in c.layers:
    r = r + c.get_region(layer)

c2 = gf.Component()
c2.add_polygon(r, layer=(1, 0))
c2
```

## Smooth

You can also simplify polygons.

```python
c = gf.components.circle()
c2 = gf.Component()
region = c.get_region(layer=LAYER.WG, smooth=1)
c2.add_polygon(region, layer=LAYER.WG)
c2
```

As well as easily do boolean operations with regions.

```python
c = gf.components.circle()
c2 = gf.Component()
region = c.get_region(layer=LAYER.WG, smooth=1)
region2 = region.sized(100)
region3 = region2 - region

c2.add_polygon(region3, layer=LAYER.WG)
c2
```

## Importing GDS files


`gf.import_gds()` allows you to easily import external GDSII files.  It imports a single cell from the external GDS file and converts it into a gdsfactory component.

```python
D = gf.components.ellipse()
D.write_gds("myoutput.gds")
D2 = gf.import_gds(gdspath="myoutput.gds")
D2.plot()
```

## Copying and extracting geometry

```python
E = gf.Component()
E.add_ref(gf.components.ellipse(layer=(1, 0)))
D = E.extract(layers=[(1, 0)])
D.plot()
```

```python
import gdsfactory as gf

X = gf.components.ellipse(layer=(2, 0))
c = X.copy()
c.plot()
```

```python
c = gf.c.straight()
c = c.copy()

 # Type: ignore to tell the static type checker, like Mypy or Pylance, to deliberately ignore a potential type error on that specific line of code.
 # In this case, the copy_layers method might be type-hinted to expect a dictionary with a more specific key type,
 # such as gf.typings.Layer, instead of a simple string like "WG". This might potentially flag as error or mismatch.
c.copy_layers(layer_map=dict(WG=(2, 0)), recursive=False)
c
```

## Import Images into GDS

You can import your logo into GDS using the conversion from numpy arrays.

```python
import gdsfactory as gf
from gdsfactory.config import PATH
from gdsfactory.read.from_np import from_image

c = from_image(
    PATH.module / "samples" / "images" / "logo.png", nm_per_pixel=500, invert=False
)
c.plot()
```

```python
c = from_image(
    PATH.module / "samples" / "images" / "logo.png", nm_per_pixel=500, invert=True
)
c.plot()
```

## Dummy Fill / Tiling

To keep constant density in some layers you can add dummy fill shapes.

For big layouts you should use a tiling processor.

```python
import gdsfactory as gf

c = gf.Component()
c << gf.components.die_with_pads()
fill = gf.c.rectangle(layer="M3")
c.fill(
    fill_cell=fill,
    fill_layers=[("FLOORPLAN", -100)],
    exclude_layers=[
        ((1, 0), 100),
        ("M3", 100)
        ],
    x_space=1,
    y_space=1,
)
c
```

```python
@gf.cell
def fill_cell():
    c = gf.Component()
    c << gf.c.rectangle(layer=(3,0), size=(1,1))
    c << gf.c.rectangle(layer=(2,0), size=(1,1))
    return c

c = gf.Component()
r = c << gf.c.ellipse(layer=(1,0), radii=(100,50))
t = c << gf.c.triangle(layer=(2,0))

c.fill(fill_cell=fill_cell(),
    fill_layers=[("WG", -1)],
    exclude_layers=[((2,0), 10)],
    x_space=1,
    y_space=1,
)
c
```
